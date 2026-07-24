
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci

def k_grid_calc(reciprocal_vectors_list,nk1,nk2):

    v1 = np.array(reciprocal_vectors_list[0])
    v2 = np.array(reciprocal_vectors_list[1])

    s = np.linspace(0, 1, nk1+1)
    t = np.linspace(0, 1, nk2+1)

    k_points = np.array([s_i * v1 + t_j * v2 for s_i in s[:-1] for t_j in t[:-1]])
    
    return k_points

def reciprocal_vectors_calc(lat):
    if lat.a3 is not None:

        V=np.dot(lat.a1,np.cross(lat.a2,lat.a3))
        reciprocal_vectors_list=[2*np.pi*np.cross(lat.a2,lat.a3)/V,
                                2*np.pi*np.cross(lat.a3,lat.a1)/V,
                                2*np.pi*np.cross(lat.a1,lat.a2)/V]
    else:
        a3=np.array([0,0,10])
        
        if len(lat.a1)==2:
            print()
            a1=np.array(list(lat.a1)+[0])
            a2=np.array(list(lat.a2)+[0])
        else:
            a1=lat.a1
            a2=lat.a2


        V=np.dot(a1,np.cross(a2,a3))
        reciprocal_vectors_list=[2*np.pi*np.cross(a2,a3)/V,
                                2*np.pi*np.cross(a3,a1)/V,
                                2*np.pi*np.cross(a1,a2)/V]

    return reciprocal_vectors_list

class Model:

    def __init__(self, lattice, n1=None, n2=None,space="reciprocal"):

        self.lattice = lattice
        self.n1=n1
        
        if n2 is None:
            n2=n1
        
        self.n2=n2

        self.space = space.lower()

        if self.space in ["k", "reciprocal"]:


            self.reciprocal_vectors=reciprocal_vectors_calc(self.lattice)

            self.b1 = self.reciprocal_vectors[0]
            self.b2 = self.reciprocal_vectors[1]
            self.b3 = self.reciprocal_vectors[2]
            


            if len(self.lattice.sublattices[0]['position'])==3:
                self.k_grid=k_grid_calc([self.b1,self.b2],n1,n2)
            elif len(self.lattice.sublattices[0]['position'])==2:
                self.k_grid=k_grid_calc([self.b1[:-1],self.b2[:-1]],n1,n2)




        elif self.space in ["real", "r"]:

            self.a1 = np.asarray(lattice.a1, dtype=float)
            self.a2 = np.asarray(lattice.a2, dtype=float)

        else:
            raise ValueError("space must be 'real', 'r', 'k' or 'reciprocal'")

    def Hk(self):
        print('CLass that returns Hk and the velocities')
        return 


    def make_k_path(self,k_points, points_per_segment=100, endpoint=True):
        """
        Create a k-path by linearly interpolating between a list of k-points.

        Parameters
        ----------
        k_points : array-like, shape (N, d)
            List of high-symmetry k-points.
        points_per_segment : int
            Number of points between consecutive k-points.
        endpoint : bool
            Whether to include the final k-point.

        Returns
        -------
        k_path : ndarray, shape (M, d)
            Interpolated k-path.
        """
        k_points = np.asarray(k_points, dtype=float)

        path = []
        for i in range(len(k_points) - 1):
            segment = np.linspace(
                k_points[i],
                k_points[i + 1],
                points_per_segment,
                endpoint=False
            )
            path.append(segment)

        if endpoint:
            path.append(k_points[-1][None, :])

        return np.vstack(path)



    def plot(self):
        if self.space in ["k", "reciprocal"]: 
            plt.quiver(0, 0,self.b1[0], self.b1[1],angles="xy",scale_units="xy",scale=1,color="gray")
            plt.quiver(0, 0,self.b2[0], self.b2[1],angles="xy",scale_units="xy",scale=1,color="gray")

            plt.xlim([np.min([self.b1[0],self.b2[0]])-0.25,np.max([self.b1[0],self.b2[0]])+0.25])
            plt.ylim([np.min([self.b1[1],self.b2[1]])-0.25,np.max([self.b1[1],self.b2[1]])+0.25])

            plt.scatter(self.k_grid[:,0],self.k_grid[:,1])
            plt.title(str(self.n1)+'x'+str(self.n2)+' = '+str(self.n1*self.n2)+' points')
    
        

    

    def hamiltonian(self):

        N = self.n1 * self.n2 * self.lattice.norbs

        H_entries = []
        Dx_entries = []
        Dy_entries = []

        S = np.zeros((N, len(self.lattice.a1)), dtype=float)
        Omega=np.linalg.norm(np.cross(self.n1*self.lattice.a1,self.n2*self.lattice.a2))

        def cell_index(ix, iy):
            return (iy * self.n1 + ix) * self.lattice.norbs

        # ----------------------------------------------------
        # Onsites + orbital positions
        # ----------------------------------------------------

        index_list = [[] for _ in range(self.lattice.norbs)]
        index_atoms=[]

        for ix in range(self.n1):
            for iy in range(self.n2):

                R = ix * self.lattice.a1 + iy * self.lattice.a2
                offset = cell_index(ix, iy)

                for sub in self.lattice.sublattices:

                    pos = R + np.asarray(sub["position"], dtype=float)
                    inds = self.lattice.sub_index_orb[sub["name"]]

                    onsite = np.asarray(sub["onsite"])

                    if onsite.ndim == 0 or onsite.ndim == 1 :
                        onsite = onsite.reshape((1, 1))



                    for i, oi in enumerate(inds):

                        
                        S[offset + oi] = pos

                        index_list[oi].append(offset + oi)
                        index_atoms.append(self.lattice.sub_index_sml[sub["name"]])


                        for j, oj in enumerate(inds):
                            H_entries.append((offset + oi,offset + oj,onsite[i, j]))
    

        # ----------------------------------------------------
        # Hoppings
        # ----------------------------------------------------
        for ix in range(self.n1):
            for iy in range(self.n2):

                offset1 = cell_index(ix, iy)

                Ri = ix * self.lattice.a1 + iy * self.lattice.a2

                for hop in self.lattice.hoppings:

                    dx, dy = hop["relative_index"]

                    jx = (ix + dx) % self.n1
                    jy = (iy + dy) % self.n2

                    offset2 = cell_index(jx, jy)

                    from_inds = self.lattice.sub_index_orb[hop["from"]]
                    to_inds = self.lattice.sub_index_orb[hop["to"]]

                    t = np.asarray(hop["energy"], dtype=np.complex128)
                    if t.ndim == 0:
                        t = t.reshape((1, 1))

                    sub_from = self.lattice.sublattices[
                        self.lattice.sub_index_sml[hop["from"]]]
                    sub_to = self.lattice.sublattices[
                        self.lattice.sub_index_sml[hop["to"]]]

                    r_from = Ri + np.asarray(sub_from["position"], dtype=float)
                    r_to = (
                        Ri
                        + dx * self.lattice.a1
                        + dy * self.lattice.a2
                        + np.asarray(sub_to["position"], dtype=float))

                    dr = r_to - r_from


                    for i, oi in enumerate(from_inds):
                        for j, oj in enumerate(to_inds):

                            H_entries.append((offset1+oi, offset2+oj, t[i,j]))
                            Dx_entries.append((offset1 + oi,offset2 + oj,  dr[0]))
                            Dy_entries.append((offset1 + oi,offset2 + oj,  dr[1]))

                            H_entries.append((offset2+oj, offset1+oi, t[i,j].conjugate()))
                            Dx_entries.append((offset2 + oj,offset1 + oi,  -dr[0]))
                            Dy_entries.append((offset2 + oj,offset1 + oi,  -dr[1]))

                            





        def build_csr(entries):
            rows = np.fromiter((e[0] for e in entries), dtype=np.int32)
            cols = np.fromiter((e[1] for e in entries), dtype=np.int32)
            data = np.fromiter((e[2] for e in entries), dtype=np.asarray(entries[0][2]).dtype)
            return sci.sparse.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()


        H = build_csr(H_entries)
        
        Dx = build_csr(Dx_entries)

        Dy = build_csr(Dy_entries)

        print('Number of unit cells : '+str(self.n1)+' x '+str(self.n2 )+' = '+str(self.n1*self.n2))
        print('NUmber of orbitals in each unit cell : '+str(self.lattice.norbs))
        print('H : '+str(H.shape))
        print('Dx shape : '+str(Dx.shape))
        print('Dy shape : '+str(Dy.shape))
        print('Size of H : '+str(((H.data.nbytes +H.indices.nbytes +H.indptr.nbytes)) / (1024**2))+' Mb')

        self.index_list=index_list
        self.index_atoms=np.array(index_atoms)
        self.H=H
        self.Dx=Dx
        self.Dy=Dy
        self.S=S
        return 


    




