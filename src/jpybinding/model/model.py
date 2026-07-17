
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

    def __init__(self, lattice, space="real",n1=None, n2=None):

        self.lattice = lattice
        self.n1=n1
        self.n2=n2
        self.space = space.lower()

        if self.space in ["k", "reciprocal"]:


            self.reciprocal_vectors=reciprocal_vectors_calc(self.lattice)

            self.b1 = self.reciprocal_vectors[0]
            self.b2 = self.reciprocal_vectors[1]
            self.b3 = self.reciprocal_vectors[2]
            

            self.k_grid=k_grid_calc([self.b1,self.b2],n1,n2)




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
    
        





    




