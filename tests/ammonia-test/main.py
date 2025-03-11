from manim import *
from manim_chemistry import *
from manimolconv import example as mlv

ammonia = Molecule(GraphMolecule).molecule_from_file(
        "../ammonia.mol", 
        ignore_hydrogens=False,
        label=True,
        numeric_label=False
    )

mc_ammonia = MCMolecule.construct_from_file(f"../ammonia.mol")

class GraphMoleculeFromMolecule(Scene):
    def construct(self):
        self.add(ammonia.scale(2))

        elementMatrix, featVec_group = mlv.init_element_features(mc_ammonia, ammonia, self)

        self.wait(2)

        # neighbors = VGroup(*[ammonia.atoms[i] for i in range(1,4)])
        # self.play(Indicate(neighbors))

        # self.wait(2)

        elementMatrix, featVec_group = mlv.once_convolve(elementMatrix, ammonia, featVec_group, self)
        self.wait(2)
        elementMatrix, featVec_group = mlv.once_convolve(elementMatrix, ammonia, featVec_group, self)


