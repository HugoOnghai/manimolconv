from manim import *
from manim_chemistry import *

import numpy as np
import networkx as nx
import copy

config.renderer = "cairo"

def construct_molecule(file=None):
    if file is None:
        raise ValueError("No file provided. Please specify a file.")
    
    molecule = Molecule(GraphMolecule).molecule_from_file(
        file, 
        ignore_hydrogens=False,
        label=True,
        numeric_label=False
    )

    # makes a manim_chemistry molecule
    mc_molecule = MCMolecule.construct_from_file(file)

    return mc_molecule, molecule


def init_element_features(mc_molecule, molecule, scene):
    # currently assuming that the indexing of mc_molecule is the same as molecule

    # getting distinct elements in the molecule
    elementlist = [atom.element.symbol for atom in mc_molecule.atoms]
    unique_elements = list(set(elementlist)) # making a list a set first removes redundancies

    # creating a matrix to store the one-hot encoding of each atom (rows = elements, columns = atoms)
    #element_features_matrix = np.zeros((len(unique_elements), len(molecule.get_atoms())))
    element_features_matrix = np.zeros((len(unique_elements), len(list(molecule.atoms))))

    # creating a one-hot encoding for each atom
    for atom_index, atom in enumerate(mc_molecule.atoms):
        element_features_matrix[unique_elements.index(atom.element.symbol), atom_index] = 1

    featVec_group = draw_element_features(element_features_matrix, molecule, scene)

    # returns the a matrix where the i-th column is the one-hot encoding of the (i+1)-th atom/vertex in the molecule/graph
    return element_features_matrix, featVec_group

def draw_element_features(element_features_matrix, molecule, scene):
    vg = VGroup()

    for atom_index, atom in enumerate(list(molecule.atoms)):
        pos = molecule.find_atom_position_by_index(atom_index + 1)

        featVec = Matrix( [[round(value, 2)] for value in element_features_matrix[:, atom_index].tolist()] ).scale(0.5)
    
        featVec.move_to(pos)
        featVec.shift(UP)

        vg.add(featVec)

    scene.play(Write(vg))

    # returns all displayed feature vectors as individual manim matrix mobjects inside a VGroup, ordered in accordance with their given atom/node molecule/graph order.
    return vg

def once_convolve(element_features_matrix, molecule, featVec_group, scene, run_time=1):
    new_element_features_matrix = copy.deepcopy(element_features_matrix)

    new_featVec_VGroup = VGroup()

    for atom_index, atom in enumerate(list(molecule.atoms)):
        curr_featVec = copy.deepcopy(element_features_matrix[:, atom_index])
        curr_featVec_MObject = featVec_group[atom_index]

        # get the neighbors of the atom
        # index is 1-based, for the graph
        neighbors_index = molecule._graph.neighbors(atom_index + 1)
        neighbors_list = list(neighbors_index)
        num_neighbors = len(neighbors_list)

        neighborVec_group = VGroup()
        for index in neighbors_list:
            neighbor_featVec = element_features_matrix[:, index - 1] # grabs values from the origianl element_features_matrix before any convolutions were done
            neighborVec_group.add(featVec_group[index-1])
            curr_featVec += neighbor_featVec

        curr_featVec /= (num_neighbors+1) # adding one to count the atom itself 

        new_element_features_matrix[:, atom_index] = curr_featVec

        # indicate atom and its neighbors
        # molecule.atoms is a dict so it is also 1-based
        animate_conv_at_atom = []
        for i in neighbors_list:
            animate_conv_at_atom.append(Indicate(molecule.atoms[i], run_time=1, color=GREEN))

        new_featVec = Matrix( [[round(value, 2)] for value in new_element_features_matrix[:, atom_index].tolist()] ).scale(0.5)
        new_featVec.move_to(curr_featVec_MObject)

        new_featVec_VGroup.add(new_featVec) # since atoms iterate in order, we can always add the new featVec to the end of the VGroup

        scene.play(Indicate(molecule.atoms[atom_index+1], run_time=1, color=RED), *animate_conv_at_atom, run_time=run_time)
        scene.play(FadeOut(curr_featVec_MObject), TransformFromCopy(neighborVec_group, new_featVec), run_time=run_time)

    return new_element_features_matrix, new_featVec_VGroup