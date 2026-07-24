import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import jpybinding as jpb


def lattice_square(t=-1):
    
    d = 1  # [nm] unit cell length

    # create a simple 2D lattice with vectors a1 and a2
    lattice = jpb.Lattice(a1=[d, 0], a2=[0, d])
    lattice.add_sublattices(
        ('A', [0, 0])  # add an atom called 'A' at position [0, 0]
    )
    lattice.add_hoppings(
        # (relative_index, from_sublattice, to_sublattice, energy)
        ([1, 0], 'A', 'A', t),
        ([0, 1], 'A', 'A', t))
    

    return lattice



def lattice_monolayer_graphene(m=0):
    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    t = -2.7      # [eV] nearest neighbour hopping

    lat = jpb.Lattice(a1=[a, 0],
                     a2=[a/2, a/2 * np.sqrt(3)])
    lat.add_sublattices(('A', [0, -a_cc/2],m),
                        ('B', [0,  a_cc/2],-m))
    lat.add_hoppings(
        # inside the main cell
        ([0,  0], 'A', 'B', t),
        # between neighboring cells
        ([1, -1], 'A', 'B', t),
        ([0, -1], 'A', 'B', t)

    )
    return lat


def lattice_monolayer_graphene_4atoms(m=0):
    """Nearest-neighbor with 4 atoms per unit cell: square lattice instead of oblique

    Parameters
    ----------
    onsite : Tuple[float, float]
        Onsite energy for sublattices A and B.
    """
    
    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance

    t=-2.7

    lat = jpb.Lattice(a1=[a, 0], a2=[0, 3*a_cc])

    lat.add_sublattices(('A1',  [  0, 0], m),
                        ('B1',  [  0,  a_cc], -m),
                        ('A2',  [a / 2, 1.5*a_cc], m),
                        ('B2',  [a / 2, 2.5 * a_cc], -m))
    lat.add_hoppings(
        # inside the unit sell
        ([0, 0], 'A1',  'B1',  t),
        ([0, 0], 'B1',  'A2', t),
        ([0, 0], 'A2', 'B2', t),
        # between neighbouring unit cells
        ([-1, -1], 'A1', 'B2', t),
        ([ 0, -1], 'A1', 'B2', t),
        ([-1,  0], 'B1', 'A2', t),
    )

    return lat


def attice_Haldane(t=-2.7,m=0,t2=2/(3*np.sqrt(3))):

    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance

    lat = jpb.Lattice(
        a1=[a, 0],
        a2=[a/2, a/2 * np.sqrt(3)]
    )

    lat.add_sublattices(
        # name and position
        ('A', [0, -a_cc/2],-m),
        ('B', [0,  a_cc/2],m)
    )

    lat.add_hoppings(
        # inside the main cell
        ([0,  0], 'A', 'B', t),
        # between neighboring cells
        ([1, -1], 'A', 'B', t),
        ([0, -1], 'A', 'B', t),

        ([1, 0], 'A', 'A', t2 * 1j),
        ([0, -1], 'A', 'A', t2 * 1j),
        ([-1, 1], 'A', 'A', t2 * 1j),

        ([1, 0], 'B', 'B', t2 * -1j),
        ([0, -1], 'B', 'B', t2 * -1j),
        ([-1, 1], 'B', 'B', t2 * -1j)
    )

    return lat

def lattice_kane_mele(t=-2.7,m=0,lambda_I=- 1/(3*np.sqrt(3)) ,Bz=0):
    
    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    
    sz=np.array([[1,0],[0,-1]])

    lat = jpb.Lattice(
        a1=[a, 0],
        a2=[a/2, a/2 * np.sqrt(3)]
    )

    lat.add_sublattices(
        # name and position
        ('A', [0, -a_cc/2],-m*np.array([[1,0],[0,1]])+np.array([[Bz,0],[0,-Bz]])),
        ('B', [0,  a_cc/2],m*np.array([[1,0],[0,1]])+np.array([[Bz,0],[0,-Bz]]))
    )

    lat.add_hoppings(
        # inside the main cell
        ([0,  0], 'A', 'B', t*np.array([[1,0],[0,1]])),
        # between neighboring cells
        ([1, -1], 'A', 'B', t*np.array([[1,0],[0,1]])),
        ([0, -1], 'A', 'B', t*np.array([[1,0],[0,1]])),
        ([1, 0], 'A', 'A', lambda_I * 1j*sz),
        ([0, -1], 'A', 'A', lambda_I * 1j*sz),
        ([-1, 1], 'A', 'A', lambda_I * 1j*sz),

        ([1, 0], 'B', 'B', lambda_I * -1j*sz),
        ([0, -1], 'B', 'B', lambda_I * -1j*sz),
        ([-1, 1], 'B', 'B', lambda_I * -1j*sz)
    )

    return lat



def Slater_Koaster(e,Vpps,Vppp):
    print('Repasar con el de abajo porque no se si este modelo está bien.')
    ### Vector normalization
    e_norm=e/np.linalg.norm(e)

    ### Building the matrix
    t_mat=np.zeros((2,2),dtype=complex)
    
    # px-px hopping
    t_mat[0,0]=e_norm[0]**2*Vpps+(1-e_norm[0]**2)*Vppp
    # px-py hopping
    t_mat[0,1]=e_norm[0]*e_norm[1]*(Vpps-Vppp)
    # py-px hopping
    t_mat[1,0]=np.conj(t_mat[0,1])
    # py-py hopping
    t_mat[1,1]=e_norm[1]**2*Vpps+(1-e_norm[1]**2)*Vppp

    return t_mat
def Rashba(e,lambda_r):
    #The orbital part is just the identity
    orbital=np.array([[1,1],[1,1]])
    # Auxiliary vector
    sigma_x=np.array([[0,1],[1,0]])
    sigma_y=np.array([[0,-1j],[1j,0]])
    spin=(sigma_x*e[1]-sigma_y*e[0])

    return 2j*lambda_r*np.kron(orbital,spin)



def lattice_bismuthene_SOC_2atoms(Vpps=1.815,Vppp=-0.315,m=0.2,lambda_i=0.435):
    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    
    # Define the lattice vectors
    a1=[a, 0]
    a2=[a/2, a/2 * np.sqrt(3)]
    # Distance of the three different that the hoppings are going to
    e1=np.array([0,a_cc])
    e2=e1+a1-a2
    e3=e1-a2
    # Intrinsic soc
    Lz=np.array([[0,-1j],[1j,0]])
    Sz=np.array([[1,0],[0,-1]])


    lat = jpb.Lattice(a1,a2)

    lat.add_sublattices(('A', [0, -a_cc/2],0.5*m*np.eye(4)+lambda_i*np.kron(Lz,Sz)),
                        ('B', [0,  a_cc/2],-0.5*m*np.eye(4)+lambda_i*np.kron(Lz,Sz)))
    lat.add_hoppings(
        # inside the main cell
        ([0,  0], 'A', 'B',np.kron(Slater_Koaster(e1,Vpps,Vppp),np.eye(2))),
        ([1, -1], 'A', 'B',np.kron(Slater_Koaster(e2,Vpps,Vppp),np.eye(2))),
        ([0, -1], 'A', 'B',np.kron(Slater_Koaster(e3,Vpps,Vppp),np.eye(2)))

    )
    return lat



def lattice_bismuthene_SOC_4atoms(Vpps=1.815,Vppp=-0.315,m=0.2,lambda_i=0.435):
    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance

    Lz=np.array([[0,-1j],[1j,0]])
    Sz=np.array([[1,0],[0,-1]])

    # Define the lattice vectors
    a1=np.array([a, 0])
    a2=np.array([0, 3*a_cc])

    lat = jpb.Lattice(a1,a2)


    pos_A1=np.array([  0, -a_cc/2])
    pos_B1=np.array([  0,  a_cc/2])
    pos_A2=np.array([a / 2, a_cc])
    pos_B2=np.array([a / 2, 2 * a_cc])
    
    lat.add_sublattices(('A1',pos_A1,0.5*m*np.eye(4)+lambda_i*np.kron(Lz,Sz)),
                        ('B1',pos_B1,-0.5*m*np.eye(4)+lambda_i*np.kron(Lz,Sz)),
                        ('A2',pos_A2,0.5*m*np.eye(4)+lambda_i*np.kron(Lz,Sz)),
                        ('B2',pos_B2,-0.5*m*np.eye(4)+lambda_i*np.kron(Lz,Sz)))

    lat.add_hoppings(
        # inside the unit sell
        ([0, 0], 'A1',  'B1', np.kron(Slater_Koaster((0*a1+0*a2)+pos_B1-pos_A1,Vpps,Vppp),np.eye(2))),
        ([0, 0], 'B1',  'A2', np.kron(Slater_Koaster((0*a1+0*a2)+pos_A2-pos_B1,Vpps,Vppp),np.eye(2))),
        ([0, 0], 'A2', 'B2',  np.kron(Slater_Koaster((0*a1+0*a2)+pos_B2-pos_A2,Vpps,Vppp),np.eye(2))),
        # # between neighbouring unit cells
        ([-1, -1], 'A1', 'B2', np.kron(Slater_Koaster((-1*a1-1*a2)+pos_B2-pos_A1,Vpps,Vppp),np.eye(2))),
        ([ 0, -1], 'A1', 'B2', np.kron(Slater_Koaster((0*a1-1*a2)+pos_B2-pos_A1,Vpps,Vppp),np.eye(2))),
        ([-1,  0], 'B1', 'A2', np.kron(Slater_Koaster((-1*a1+0*a2)+pos_A2-pos_B1,Vpps,Vppp),np.eye(2))),
    )
    return lat


def lattice_kane_mele(lambda_I=- 1/(3*np.sqrt(3)),m=0,Bz=0 ):

    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    t = -2.7      # [eV] nearest neighbour hopping


    sz=np.array([[1,0],[0,-1]])

    lat = jpb.Lattice(
        a1=[a, 0],
        a2=[a/2, a/2 * np.sqrt(3)]
    )

    lat.add_sublattices(
        # name and position
        ('A', [0, -a_cc/2],-m*np.array([[1,0],[0,1]])+np.array([[Bz,0],[0,-Bz]])),
        ('B', [0,  a_cc/2],m*np.array([[1,0],[0,1]])+np.array([[Bz,0],[0,-Bz]]))
    )

    lat.add_hoppings(
        # inside the main cell
        ([0,  0], 'A', 'B', t*np.array([[1,0],[0,1]])),
        # between neighboring cells
        ([1, -1], 'A', 'B', t*np.array([[1,0],[0,1]])),
        ([0, -1], 'A', 'B', t*np.array([[1,0],[0,1]])),
        ([1, 0], 'A', 'A', lambda_I * 1j*sz),
        ([0, -1], 'A', 'A', lambda_I * 1j*sz),
        ([-1, 1], 'A', 'A', lambda_I * 1j*sz),

        ([1, 0], 'B', 'B', lambda_I * -1j*sz),
        ([0, -1], 'B', 'B', lambda_I * -1j*sz),
        ([-1, 1], 'B', 'B', lambda_I * -1j*sz)
    )

    return lat

### TMD models
tmd_params = {  # from https://doi.org/10.1103/PhysRevB.88.085433
    # ->           a,  eps1,  eps2,     t0,    t1,    t2,   t11,   t12,    t22
    "HsNz":  [0.3190, 1.046, 2.104, -0.184, 0.000, 0.000, 0.218, 0.338,  0.057,0.073*15.0],
    "NoIs":  [0.3190, 1.046, 2.104, -0.184, 0.000, 0.000, 0.218, 0.000,  0.057,0.073*15.0],
    "NoT2":  [0.3190, 1.046, 2.104, -0.184, 0.401, 0.000, 0.218, 0.000,  0.057,0.073*15.0],
    "HsMo":  [0.3190, 1.046, 2.104, -0.184, 0.401, 0.507, 0.218, 0.338,  0.057,0.073*15.0],
    "MoS2":  [0.3190, 1.046, 2.104, -0.184, 0.401, 0.507, 0.218, 0.338,  0.057,0.073],
    "WS2":   [0.3191, 1.130, 2.275, -0.206, 0.567, 0.536, 0.286, 0.384, -0.061,0.211],
    "MoSe2": [0.3326, 0.919, 2.065, -0.188, 0.317, 0.456, 0.211, 0.290,  0.130,0.091],
    "WSe2":  [0.3325, 0.943, 2.179, -0.207, 0.457, 0.486, 0.263, 0.329,  0.034,0.228],
    "MoTe2": [0.3557, 0.605, 1.972, -0.169, 0.228, 0.390, 0.207, 0.239,  0.252,0.107],
    "WTe2":  [0.3560, 0.606, 2.102, -0.175, 0.342, 0.410, 0.233, 0.270,  0.190,0.237],
    
}


def lattice_monolayer_3band_soc(name, lsoc=None,override_params=None):
    params = tmd_params.copy()
    if override_params:
        params.update(override_params)
    if lsoc is None:
        a, eps1, eps2, t0, t1, t2, t11, t12, t22,lsoc = params[name]
    else:
        a, eps1, eps2, t0, t1, t2, t11, t12, t22,_ = params[name]
    
    rt3 = np.sqrt(3)  # convenient constant

    lattice = jpb.Lattice(a1=[a, 0],a2=[1./2 * a, rt3/2 * a])



    ##setting the onsite matrices
    Energies = np.diag([eps1,eps2,eps2])
    Energies = Energies.astype(complex)

    onsite = np.kron(np.eye(2),Energies)

    HSOC =np.array([[0,0,0],
                    [0,0,2*1j],
                    [0,-2*1j,0]])
    lsoc=0.5*lsoc
    sigmaz = np.array([[0.5,0],[0,-0.5]],dtype=complex)

    onsite += np.kron(sigmaz,lsoc*HSOC)

    
    lattice.add_sublattices(('A', [0, 0],onsite))

    h1 = [[ t0, -t1,   t2],
          [ t1, t11, -t12],
          [ t2, t12,  t22]]
    h1s = np.kron(np.eye(2),h1)

    h2 = [[                    t0,     1/2 * t1 + rt3/2 * t2,     rt3/2 * t1 - 1/2 * t2],
          [-1/2 * t1 + rt3/2 * t2,     1/4 * t11 + 3/4 * t22, rt3/4 * (t11 - t22) - t12],
          [-rt3/2 * t1 - 1/2 * t2, rt3/4 * (t11 - t22) + t12,     3/4 * t11 + 1/4 * t22]]

    h2s = np.kron(np.eye(2),h2)
    h3 = [[                    t0,    -1/2 * t1 - rt3/2 * t2,     rt3/2 * t1 - 1/2 * t2],
          [ 1/2 * t1 - rt3/2 * t2,     1/4 * t11 + 3/4 * t22, rt3/4 * (t22 - t11) + t12],
          [-rt3/2 * t1 - 1/2 * t2, rt3/4 * (t22 - t11) - t12,     3/4 * t11 + 1/4 * t22]]
    h3s = np.kron(np.eye(2),h3)


    lattice.add_hoppings(([1,  0],'A','A',h1s),
                     ([0, -1],'A','A', h2s),
                     ([1, -1],'A','A', h3s))

    return lattice





def Slater_Koaster_s_px_py_pz(e,Vsss,Vsps,Vpps,Vppp):

    ### Vector normalization
    e_norm=e/np.linalg.norm(e)

    ### Building the matrix
    t_mat=np.zeros((4,4),dtype=complex)
    
    ### s-s
    t_mat[0,0]=Vsss
    ### s-p
    # s-px hopping
    t_mat[0,1]=e_norm[0]*Vsps
    t_mat[1,0]=-e_norm[0]*Vsps
    # s-py hopping
    t_mat[0,2]=e_norm[1]*Vsps
    t_mat[2,0]=-e_norm[1]*Vsps
    # s-pz hopping
    t_mat[0,3]=e_norm[2]*Vsps
    t_mat[3,0]=-e_norm[2]*Vsps
            
    ### p-p


    # px-px hopping
    t_mat[1,1]=(e_norm[0]**2*Vpps+(1-e_norm[0]**2)*Vppp)
    # px-py hopping
    t_mat[1,2]=e_norm[0]*e_norm[1]*(Vpps-Vppp)
    # px-pz hopping
    t_mat[1,3]=e_norm[0]*e_norm[2]*(Vpps-Vppp)

    # py-py hopping
    t_mat[2,2]=(e_norm[1]**2*Vpps+(1-e_norm[1]**2)*Vppp)
    # py-px hopping
    t_mat[2,1]=e_norm[0]*e_norm[1]*(Vpps-Vppp)
    # py-pz hopping
    t_mat[2,3]=e_norm[1]*e_norm[2]*(Vpps-Vppp)
    
    # pz-pz hopping
    t_mat[3,3]=(e_norm[2]**2*Vpps+(1-e_norm[2]**2)*Vppp)
    # pz-px hopping
    t_mat[3,1]=e_norm[0]*e_norm[2]*(Vpps-Vppp)
    # pz-py hopping
    t_mat[3,2]=e_norm[1]*e_norm[2]*(Vpps-Vppp)


    return t_mat



def lattice_square_SK_SOC(Es=3.2,Ep=-0.5,Vsss=-0.5,Vsps=0.5,Vpps=0.5,Vppp=-0.2,lambda_SOC=0.1):
    """
    Default parameters are extracted from : https://doi.org/10.1103/PhysRevLett.121.086602 (and matching with some Tatiana paper)

    """

    
    d = 2  # [A] unit cell length
    pos_A=np.array([0, 0, 0])
    a1=np.array([d, 0, 0])
    a2=np.array([0, d, 0])
    onsites=np.diag([Es,Ep,Ep,Ep])
    # SOC part
    Lx=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,-1j],[0,0,1j,0]])
    Ly=np.array([[0,0,0,0],[0,0,0,1j],[0,0,0,0],[0,-1j,0,0]])
    Lz=np.array([[0,0,0,0],[0,0,-1j,0],[0,1j,0,0],[0,0,0,0]])

    Sx=0.5*np.array([[0,1],[1,0]])
    Sy=0.5*np.array([[0,-1j],[1j,0]])
    Sz=0.5*np.array([[1,0],[0,-1]])
    L_S=2*lambda_SOC*(np.kron(Lx,Sx)+np.kron(Ly,Sy)+np.kron(Lz,Sz))

    
    
    # create a simple 2D lattice with vectors a1 and a2
    lattice = jpb.Lattice(a1, a2)
    lattice.add_sublattices(
        ('A', pos_A ,np.kron(onsites,np.eye(2))+L_S),  # add an atom called 'A' at position [0, 0]
    )
    lattice.add_hoppings(
        # (relative_index, from_sublattice, to_sublattice, energy)
        ## Same lattice
        ([1, 0], 'A', 'A', np.kron(Slater_Koaster_s_px_py_pz((1*a1+0*a2)+pos_A-pos_A,Vsss,Vsps,Vpps,Vppp),np.eye(2))),
        ([0, 1], 'A', 'A', np.kron(Slater_Koaster_s_px_py_pz((0*a1+1*a2)+pos_A-pos_A,Vsss,Vsps,Vpps,Vppp),np.eye(2))), 
 )
    

    return lattice


def lattice_square_bipartite_SK_SOC(m=0,Es=3.2,Ep=-0.5,Vsss=-0.5,Vsps=0.5,Vpps=0.5,Vppp=-0.2,lambda_SOC=0.1,delta=0.5):
    """                                
    Default parameters are extracted from : https://doi.org/10.1103/PhysRevLett.121.086602 (and match them with some tatiana paper)

    """

    
    d = 2  # [A] unit cell length
    pos_A=np.array([0, 0, 0])
    pos_B=np.array([d/2, d/2,0])

    a1=np.array([d, 0, 0])
    a2=np.array([0, d, 0])
    onsites=np.diag([Es,Ep,Ep,Ep])
    # SOC part
    Lx=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,-1j],[0,0,1j,0]])
    Ly=np.array([[0,0,0,0],[0,0,0,1j],[0,0,0,0],[0,-1j,0,0]])
    Lz=np.array([[0,0,0,0],[0,0,-1j,0],[0,1j,0,0],[0,0,0,0]])

    Sx=0.5*np.array([[0,1],[1,0]])
    Sy=0.5*np.array([[0,-1j],[1j,0]])
    Sz=0.5*np.array([[1,0],[0,-1]])
    L_S=2*lambda_SOC*(np.kron(Lx,Sx)+np.kron(Ly,Sy)+np.kron(Lz,Sz))

    
    
    # create a simple 2D lattice with vectors a1 and a2
    lattice = jpb.Lattice(a1, a2)
    lattice.add_sublattices(
        ('A', pos_A ,m*np.kron(np.eye(4),np.eye(2))+np.kron(onsites,np.eye(2))+L_S),  # add an atom called 'A' at position [0, 0]
        ('B', pos_B ,-m*np.kron(np.eye(4),np.eye(2))+np.kron(onsites,np.eye(2))+L_S),  # add an atom called 'A' at position [0, 0]
    )
    lattice.add_hoppings(
        # (relative_index, from_sublattice, to_sublattice, energy)
        ## Same lattice
        ([1, 0], 'A', 'A', np.kron(Slater_Koaster_s_px_py_pz((1*a1+0*a2)+pos_A-pos_A,Vsss,Vsps,Vpps,Vppp),np.eye(2))),
        ([0, 1], 'A', 'A', np.kron(Slater_Koaster_s_px_py_pz((0*a1+1*a2)+pos_A-pos_A,Vsss,Vsps,Vpps,Vppp),np.eye(2))), 
        ([1, 0], 'B', 'B', np.kron(Slater_Koaster_s_px_py_pz((1*a1+0*a2)+pos_B-pos_B,Vsss,Vsps,Vpps,Vppp),np.eye(2))),
        ([0, 1], 'B', 'B', np.kron(Slater_Koaster_s_px_py_pz((0*a1+1*a2)+pos_B-pos_B,Vsss,Vsps,Vpps,Vppp),np.eye(2))),
        ## Different lattice
        ([0, 0], 'A', 'B',  np.kron(delta*Slater_Koaster_s_px_py_pz((+0*a1+0*a2)+pos_B-pos_A,Vsss,Vsps,Vpps,Vppp),np.eye(2))),
        ([-1, -1], 'A', 'B',  np.kron(delta*Slater_Koaster_s_px_py_pz((+1*a1+1*a2)+pos_B-pos_A,Vsss,Vsps,Vpps,Vppp),np.eye(2))),
        ([-1, 0], 'A', 'B', np.kron(delta*Slater_Koaster_s_px_py_pz((-1*a1+0*a2)+pos_B-pos_A,Vsss,Vsps,Vpps,Vppp),np.eye(2))),
        ([0, -1], 'A', 'B', np.kron(delta*Slater_Koaster_s_px_py_pz((+0*a1-1*a2)+pos_B-pos_A,Vsss,Vsps,Vpps,Vppp),np.eye(2))),
        
 )
    

    return lattice