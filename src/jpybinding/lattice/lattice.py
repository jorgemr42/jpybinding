import numpy as np
import matplotlib.pyplot as plt

class Lattice:

    def __init__(self, a1, a2,a3=None):

        self.a1 = np.asarray(list(a1))
        self.a2 = np.asarray(list(a2))
        
        if a3 is not None:
            self.a3 = np.asarray(list(a3))
        else:
            self.a3=None
        
        self.sublattices = []
        self.hoppings = []

        # Diccionario interno: nombre -> índice
        self.sub_index_sml = {}
        self.sub_index_orb = {}

    def add_sublattices(self, *sublattices):
        """
        Añade una o varias subredes.

        Ejemplo
        --------
        lattice.add_sublattices(
            ("A", [0, 0], onsite),
            ("B", [0.5, 0.5], onsite)
        )
        """
    def add_sublattices(self, *sublattices):
        sub_index=0
        for sub in sublattices:

            if len(sub) == 2:
                name, position = sub
                onsite = None

            elif len(sub) == 3:
                name, position, onsite = sub

            else:
                raise ValueError(
                    "Each sublattice must be (name, position) or (name, position, onsite)."
                )

            if name in self.sub_index_sml:
                raise ValueError(f"Sublattice '{name}' already exists.")

            self.sub_index_sml[name] = len(self.sublattices)


            if onsite is not None:
                self.sub_index_orb[name]=[]
                if np.isscalar(onsite):
                    onsite = np.asarray([onsite])
                for i in range(np.asarray(onsite).shape[0]):
                    self.sub_index_orb[name].append(sub_index)
                    sub_index+=1
                onsite = np.asarray(onsite).astype(np.complex128)

            else:
                self.sub_index_orb[name] = [sub_index]
                sub_index+=1

                onsite = np.array([0j])
            


            self.sublattices.append({
                "name": name,
                "position": list(position),
                "onsite": onsite
            })
        self.norbs=sub_index
        return self

    def add_hoppings(self, *hoppings):
        """
        Añade uno o varios hoppings.

        Ejemplo
        --------
        lattice.add_hoppings(
            ([1, 0], "A", "B", -1),
            ([0, 1], "A", "B", -1)
        )
        """

        for relative_index, from_name, to_name, energy in hoppings:

            if from_name not in self.sub_index_sml:
                raise ValueError(f"Unknown sublattice '{from_name}'.")

            if to_name not in self.sub_index_sml:
                raise ValueError(f"Unknown sublattice '{to_name}'.")

            self.hoppings.append({
                "relative_index": list(relative_index),
                "from": from_name,
                "to": to_name,
                "energy": np.asarray(energy).astype(np.complex128)
            })

        return self
    
    def plot(self):
        """
        Plot the unit cell, sublattices, lattice vectors and hoppings.
        """
        ### Lattice vectors :

        a1 = np.asarray(self.a1)
        a2 = np.asarray(self.a2)

        plt.quiver(0, 0,a1[0], a1[1],angles="xy",scale_units="xy",scale=1,color="gray")
        plt.quiver(0, 0,a2[0], a2[1],angles="xy",scale_units="xy",scale=1,color="gray")

        ## Sublattices

        for i in range(len(self.sublattices)):
            if len(self.sublattices[i]["position"])==2:
                x, y = self.sublattices[i]["position"]
            elif len(self.sublattices[i]["position"])==3:
                x, y,_ = self.sublattices[i]["position"]
            plt.scatter(x, y,s=1000)
            plt.text(x,y,self.sublattices[i]["name"],fontsize=12,ha='center',va='center',bbox=dict(facecolor="lightgray",alpha=0.6,edgecolor="none",boxstyle="round,pad=0.2"))



        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True, which="both", linestyle="--", alpha=0.4,zorder=0)

        plt.xlim([np.min([a1[0],a2[0]])-0.25,np.max([a1[0],a2[0]])+0.25])
        plt.ylim([np.min([a1[1],a2[1]])-0.25,np.max([a1[1],a2[1]])+0.25])