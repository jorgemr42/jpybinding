import numpy as np
import scipy as sci


class Solver:

    def __init__(self, model):

        self.model = model
     




    def H_k_1(self,k_point=None):
        
        Hk_list=[]
        for sub in self.model.lattice.sublattices:
        
            Hk_list.append(sub['onsite'])
        
        Hk=sci.linalg.block_diag(*Hk_list)

        for hop in self.model.lattice.hoppings:
            
            rows = self.model.lattice.sub_index_orb[hop['from']]
            cols = self.model.lattice.sub_index_orb[hop['to']]

            R = (hop['relative_index'][0] * self.model.lattice.a1 +
                hop['relative_index'][1] * self.model.lattice.a2)

            r_from = self.model.lattice.sublattices[
                self.model.lattice.sub_index_sml[hop['from']]
            ]['position']

            r_to = self.model.lattice.sublattices[
                self.model.lattice.sub_index_sml[hop['to']]
            ]['position']

            dist = np.asarray(r_from) - np.asarray(r_to) - np.asarray(R)
            value=hop['energy']*np.exp(-1j*(np.dot(k_point,dist)))

            Hk[np.ix_(rows, cols)] += value
            Hk[np.ix_(cols, rows)] += value.conj().T



        return Hk




    def calc_bands(self,k_points=None,k_points_labels=None,points_per_segment=100):
        
        k_path=self.model.make_k_path(k_points, points_per_segment, endpoint=True)
    
        ene_mat=np.zeros((len(k_path),self.model.lattice.norbs),dtype=np.complex128)
    
        for i in range(len(k_path)):
            Hk=self.H_k_1(k_path[i])
            ene_mat[i,:],_=np.linalg.eigh(Hk)
        return ene_mat
            









    

 