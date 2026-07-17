import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

class Solver:

    def __init__(self, model):

        self.model = model
        self.ene=None
        self.vec=None





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


    def H_k_1_and_V_k_1(self,k_point=None):
        
        Hk_list=[]
        for sub in self.model.lattice.sublattices:
        
            Hk_list.append(sub['onsite'])
        
        Hk=sci.linalg.block_diag(*Hk_list)
        Vk1=np.zeros(Hk.shape,dtype=np.complex128)
        Vk2=np.zeros(Hk.shape,dtype=np.complex128)

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



    def calc_bands(self,k_points=None,plot=True,k_points_labels=None,k_points_per_segment=100):
        
        k_path=self.model.make_k_path(k_points, k_points_per_segment, endpoint=True)
        

        ene_mat=np.zeros((len(k_path),self.model.lattice.norbs),dtype=np.float64)
        vec_mat=np.zeros((len(k_path),self.model.lattice.norbs,self.model.lattice.norbs),dtype=np.complex128)
    
        for i in range(len(k_path)):
            Hk=self.H_k_1(k_path[i])
            ene_mat[i,:],vec_mat[i,:]=np.linalg.eigh(Hk)

        self.ene=ene_mat
        self.vec=vec_mat
        self.k_path=k_path
        self.k_points_labels=k_points_labels
        self.k_points_per_segment=k_points_per_segment

        if plot is True:

            for i in range(self.model.lattice.norbs):
                plt.plot(ene_mat[:,i],color='tab:blue')
            
            if k_points_labels is not None:
                for i in range(len(k_points_labels)):
                    plt.axvline(x=i*k_points_per_segment,linestyle='dashed',color='gray',alpha=0.5)

                plt.xticks(list(range(0,len(k_points_labels)*k_points_per_segment,k_points_per_segment)),labels=k_points_labels)


            plt.ylabel('Energy [eV]')
            

    def calc_polarization(self,operator_list,Temp=300,plot=True,operator_label_list=None,colorbar_list=None,k_points=None,k_points_labels=None,k_points_per_segment=100,e_r=0):

        if self.ene is None:
            self.calc_bands(k_points,False,k_points_labels,k_points_per_segment)

        kb=8.617333*1e-5 #in eV/K^-1

        pol_mat=np.zeros((len(operator_list),len(self.k_path),self.model.lattice.norbs),dtype=np.complex128)
        pol_mat_filter=np.zeros((len(operator_list),len(self.k_path),self.model.lattice.norbs),dtype=np.complex128)

        for i in range(len(self.k_path)):
            for j in range(self.model.lattice.norbs):
                for p in range(len(operator_list)):
                    pol_mat[p,i,j]=np.vdot(self.vec[i,:,j],operator_list[p]@self.vec[i,:,j])
        
            for j in range(self.model.lattice.norbs):
                for j2 in range(self.model.lattice.norbs):
                    delta_e = self.ene[i,j2] - self.ene[i,j]
                    kernel = 2/(1+np.cosh(delta_e/(kb*Temp)))
                    for p in range(len(operator_list)):
                        pol_mat_filter[p,i,j] += pol_mat[p,i,j2] * kernel

            
        if plot is True:

            if colorbar_list is None:
                colorbar_list=[]
                for i in range(len(operator_label_list)):
                    colorbar_list.append('coolwarm')



            for p in range(len(operator_label_list)):
                operator_label=operator_label_list[p]
                idx_operator=p
                
                colorbar_color=colorbar_list[p]
                

                v_max=(np.max((np.real(pol_mat_filter[idx_operator,:]))))
                v_min=(np.min((np.real(pol_mat_filter[idx_operator,:]))))
        

                plt.figure(figsize=(7,7))

                #Plot of the bands projected on the mean value
                for j in range(self.ene.shape[1]):
                    plt.scatter(np.array(range(len(self.k_path))),self.ene[:,j]-e_r,c=np.real(pol_mat_filter[idx_operator,:,j]),cmap=colorbar_color,vmin=v_min,vmax=v_max,zorder=2,s=9)

                if self.k_points_labels is not None:
                    for i in range(len(self.k_points_labels)):
                        plt.axvline(x=i*self.k_points_per_segment,linestyle='dashed',color='gray',alpha=0.5,zorder=0)

                    plt.xticks(list(range(0,len(self.k_points_labels)*self.k_points_per_segment,self.k_points_per_segment)),labels=self.k_points_labels)

                plt.ylabel('Energy [eV]')
                plt.colorbar(label=operator_label)
                plt.show()

        return pol_mat,pol_mat_filter

    


























    
            









    

 