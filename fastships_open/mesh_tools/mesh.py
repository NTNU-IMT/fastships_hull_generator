# © 2023, NTNU
# Author: John Martin Kleven Godø <john.martin.godo@ntnu.no>
# This code is licenced under the GNU General Public License v3.0

import numpy as np
from numba import jit, njit, prange


def Rzyx(phi, theta, psi):
	'''Create rotation matrix for Euler angles phi, theta and psi-
	See 2011, Fossen, T.I. Handbook of Marine Craft Hydrodynamics and Motion Control
	for details'''
	Rbn 		= np.zeros((3,3))
	Rbn[0,0] 	= np.cos(psi)*np.cos(theta)
	Rbn[0,1] 	= -np.sin(psi)*np.cos(phi) + np.cos(psi)*np.sin(theta)*np.sin(phi)
	Rbn[0,2] 	= np.sin(psi)*np.sin(phi) + np.cos(psi)*np.cos(phi)*np.sin(theta)
	Rbn[1,0] 	= np.sin(psi)*np.cos(theta)
	Rbn[1,1] 	= np.cos(psi)*np.cos(phi) + np.sin(phi)*np.sin(theta)*np.sin(psi)
	Rbn[1,2] 	= -np.cos(psi)*np.sin(phi) + np.sin(theta)*np.sin(psi)*np.cos(phi)
	Rbn[2,0] 	= -np.sin(theta)
	Rbn[2,1] 	= np.cos(theta)*np.sin(phi)
	Rbn[2,2] 	= np.cos(theta)*np.cos(phi)

	return Rbn

@njit()
def calc_tri_surf_prop(verts_array):
	v1 							= verts_array[1, :] - verts_array[0, :]
	v2 							= verts_array[2, :] - verts_array[0, :]
	cross_p_vec 				= np.cross(v1, v2)
	cross_p_vec_abs 			= np.sqrt(np.sum(cross_p_vec**2))
	A 							= 0.5 * cross_p_vec_abs
	if cross_p_vec_abs != 0:
		n_vec 						= cross_p_vec/cross_p_vec_abs
	else:
		n_vec 						= np.zeros(3)
	return A, n_vec

@njit()
def calc_face_properties(faces_array, verts_array, verts_per_face_vec, n_faces):
	print('Calculating centers, areas and normal vectors of faces.')
	#n_faces 								= np.shape(faces_array)[0]
	face_centers 							= np.zeros((n_faces, 3))
	face_areas								= np.zeros(n_faces)
	face_n_vecs								= np.zeros((n_faces, 3))
	for i in range(n_faces):
		face_verts 							= verts_array[faces_array[i, :].astype('int')]
		for j in range(3):
			face_centers[i, j]					= np.mean(face_verts[:, j])
		# Splitting face into triangular subfaces, calculate properties and averaging/summing to get properties for the unsplitted face.
		n_tri_subfaces 						= int(verts_per_face_vec[i]) - 2
		A_tri_subfaces 						= np.zeros(n_tri_subfaces)
		n_vec_tri_subfaces 					= np.zeros((n_tri_subfaces, 3))
		for j in range(n_tri_subfaces):
			subface_verts 						= np.zeros((3, 3))
			subface_verts[0, :]					= face_verts[0, :]
			subface_verts[1, :]					= face_verts[1 + j, :]
			subface_verts[2, :]					= face_verts[2 + j, :]
			A_tri_subfaces[j], n_vec_tri_subfaces[j] 	= calc_tri_surf_prop(subface_verts)
		A_face 								= np.sum(A_tri_subfaces)
		n_vec_face 							= np.zeros(3)
		if A_face != 0:
			for subface_ind in range(n_tri_subfaces):
				for j in range(3):
					n_vec_face[j] 						+= A_tri_subfaces[subface_ind] / A_face * n_vec_tri_subfaces[subface_ind, j] #Area weighted average
		else:
			n_vec_face 							= np.zeros(3)
		face_areas[i] 						= A_face
		face_n_vecs[i]						= n_vec_face
	print('Finished calculating centers, areas and normal vectors of faces.')
	return face_centers, face_areas, face_n_vecs



""" def remove_zero_area_faces_jit(faces_array, face_areas, face_group_inds, verts_per_face_vec, n_faces, multiple_face_groups):
	'''Not yet working'''
	n_zero_area_faces 						= int(np.count_nonzero(face_areas == 0))
	n_faces_new 							= int(int(n_faces) - n_zero_area_faces)
	print('Removing {:d} zero area faces'.format(n_zero_area_faces))
	new_faces 								= np.zeros((n_faces_new, int(np.max(verts_per_face_vec)))).astype('int')
	new_verts_per_face_vec 					= np.zeros(n_faces_new).astype('int')
	new_face_group_inds 					= np.zeros(n_faces_new).astype('int')
	counter_new_faces 						= 0
	for i in range(n_faces):
		if face_areas[i] 						!= 0:
			n_verts_current_face 										= int(verts_per_face_vec[i]) #len(self.faces_array[i])
			for verts_ind in range(n_verts_current_face):
				new_faces[counter_new_faces, verts_ind] 		= faces_array[i, verts_ind]
			new_verts_per_face_vec[counter_new_faces] 					= n_verts_current_face
			if multiple_face_groups:
				new_face_group_inds[counter_new_faces] 						= int(face_group_inds[i])
			counter_new_faces											+= 1
	faces_array 							= new_faces.astype('int')
	verts_per_face_vec 						= new_verts_per_face_vec.astype('int')
	if multiple_face_groups:
		face_group_inds 						= new_face_group_inds.astype('int')
	n_faces 								= int(np.copy(n_faces_new))
	return faces_array, face_areas, face_group_inds, verts_per_face_vec, n_faces  """

@njit()
def identify_unused_verts_jit(faces_array, verts_array, n_verts):
	'''For all vertex indices which have a reference in faces_array: set 
	do_not_remove_verts_decision_array[vertex_index] to 1, to indicate 
	that this vertex is indeed in use. '''
	do_not_remove_verts_decision_array  	= np.zeros(n_verts).astype('int')
	for j in range(np.shape(faces_array)[0]):
		for m in range(len(faces_array[j])):
			vert_index 			= faces_array[j, m]
			do_not_remove_verts_decision_array[vert_index] 	= 1
	#for i in range(n_verts):
	#	for j in range(np.shape(faces_array)[0]):
	#		for m in range(len(faces_array[j])):
	#			if faces_array[j, m] == i:
	#				do_not_remove_verts_decision_array[i] = 1
		#if i in faces_array:
		#	do_not_remove_verts_decision_array[i] 	= 1
	return do_not_remove_verts_decision_array

@njit()
def remove_unused_verts_from_arrays_jit(faces_array, verts_array, n_verts, n_used_verts, do_not_remove_verts_decision_array):
	new_verts_array							= np.zeros((n_used_verts, 3))
	new_verts_array_counter 				= 0
	for i in range(n_verts):
		if do_not_remove_verts_decision_array[i] == 1:
			# Add vertex to new_verts_array, at its new index new_verts_array_counter
			new_verts_array[new_verts_array_counter] 	= verts_array[i]
			# In faces_array: replace all entries of the original vertex index (i) by the new vertex index (new_verts_array_counter)
			for face_ind in range(np.shape(faces_array)[0]):
				for vert_ind_in_face in range(len(faces_array[face_ind])):
					if faces_array[face_ind, vert_ind_in_face] == i:
						faces_array[face_ind, vert_ind_in_face] 	= new_verts_array_counter
			new_verts_array_counter 					+= 1
	verts_array 							= new_verts_array
	n_verts 								= n_used_verts
	return faces_array, verts_array, n_verts


class Mesh():
	def __init__(self):
		self.multiple_face_groups				= False

	def translate(self, translation_vec):
		for i in range(self.n_verts):
			self.verts_array[i] 		+= translation_vec

	def mirror(self, axis_no):
		# Create mirrored vertices
		self.mirrored_verts_array				= np.copy(self.verts_array)
		self.mirrored_verts_array[:, axis_no]	*= -1
		self.verts_array 						= np.concatenate((self.verts_array, self.mirrored_verts_array), axis = 0)
		self.n_verts 							*= 2

		# Create new faces
		self.mirrored_faces_array 				= np.copy(self.faces_array)
		self.mirrored_faces_array 				+= int(0.5*self.n_verts)
		self.mirrored_faces_array 				= np.flip(self.mirrored_faces_array, axis = 1) #Reverse order of verts in face to keep normal facing out or in as in the original face
		self.n_faces 							*= 2
		self.faces_array 						= np.concatenate((self.faces_array, self.mirrored_faces_array), axis = 0)
		self.verts_per_face_vec 				= np.concatenate((self.verts_per_face_vec, self.verts_per_face_vec), axis = 0)
		self.verts_per_face_vec					= self.verts_per_face_vec.astype('int')
		self.mirrored_face_group_inds			= np.copy(self.face_group_inds)
		self.face_group_inds					= np.concatenate((self.face_group_inds, self.mirrored_face_group_inds))

	def reverse_normal_direction(self):
		self.faces_array 						= np.flip(self.faces_array, axis = 1)


	def scale(self, scale_vector, origin):
		for i in range(self.n_verts):
			for j in range(3):
				self.verts_array[i, j]						= (self.verts_array[i, j] - origin[j]) * scale_vector[j] + origin[j]


	def calc_face_properties(self):
		self.face_centers, self.face_areas, self.face_n_vecs 	= calc_face_properties(self.faces_array, self.verts_array, self.verts_per_face_vec, self.n_faces)


	def calc_volume_properties(self, recalc_face_properties = True):
		if recalc_face_properties:
			self.calc_face_properties()
		self.volume 							= 0
		vol_center_arm 							= np.zeros(3)
		normal_vector 							= np.array((0, 0, 1)) #Normal vector on plane onto which faces are to be extruded
		extrusion_axis 							= 2
		#Extruding all faces normal to z = 0 plane and combining enclosed volumes. Negative values will be added for faces which face away from the z = 0 plane.
		for i in range(self.n_faces):
			if self.face_n_vecs[i][extrusion_axis] > 0:
				normal_pos_multiplier 					= -1 #Assuming normal vectors so point into the enclosed mesh volume.<
			else:
				normal_pos_multiplier 					= 1
			normal_vec_product 						= np.dot(self.face_n_vecs[i], normal_vector)
			projected_area 							= self.face_areas[i] * np.sqrt(np.sum(normal_vec_product**2))
			#print(self.face_n_vecs[i])
			#print(projected_area)
			#print(normal_pos_multiplier * np.sqrt(np.sum(np.dot(self.face_n_vecs[i], -normal_vector)**2)))
			extr_vol 								= projected_area * normal_pos_multiplier * self.face_centers[i, extrusion_axis]
			extr_center 							= np.copy(self.face_centers[i])
			extr_center[extrusion_axis]				*= 0.5 #In the middle between the face and the plane onto which it is extruded
			self.volume 							+= extr_vol
			vol_center_arm 							+= extr_vol * extr_center
		if self.volume != 0:
			self.volume_center 						= vol_center_arm / self.volume
		else:
			self.volume_center 						= np.zeros(3)

	def calc_surface_area(self, recalc_face_properties = True):
		''' Calculate the surface area of the mesh '''
		if recalc_face_properties:
			self.calc_face_properties()
		self.surface_area 							= np.sum(self.face_areas)

	def calc_surface_area_center(self, recalc_face_properties = True, recalc_surface_area = True):
		if recalc_face_properties:
			self.calc_face_properties()
		if recalc_surface_area:
			self.calc_surface_area(recalc_face_properties = recalc_face_properties)
		self.surface_area_center 					= np.zeros(3)
		self.surface_area_center[0]					= np.sum(self.face_areas * self.face_centers[:, 0]) / self.surface_area
		self.surface_area_center[1]					= np.sum(self.face_areas * self.face_centers[:, 1]) / self.surface_area
		self.surface_area_center[2]					= np.sum(self.face_areas * self.face_centers[:, 2]) / self.surface_area

	def calc_volume_and_area_properties(self):
		self.calc_face_properties()
		self.calc_volume_properties(recalc_face_properties = False)
		self.calc_surface_area(recalc_face_properties = False)
		self.calc_surface_area_center(recalc_face_properties = False, recalc_surface_area = False)

	def calc_dimensions_n_extremities(self):
		self.max_x 									= np.amax(self.verts_array[self.faces_array.flatten()][:, 0])
		self.min_x 									= np.amin(self.verts_array[self.faces_array.flatten()][:, 0])
		self.max_y 									= np.amax(self.verts_array[self.faces_array.flatten()][:, 1])
		self.min_y 									= np.amin(self.verts_array[self.faces_array.flatten()][:, 1])
		self.max_z 									= np.amax(self.verts_array[self.faces_array.flatten()][:, 2])
		self.min_z 									= np.amin(self.verts_array[self.faces_array.flatten()][:, 2])

		self.dim_x 									= self.max_x - self.min_x
		self.dim_y									= self.max_y - self.min_y
		self.dim_z 									= self.max_z - self.min_z

	def chop_at_z0(self, close_top = True, split_to_face_groups = False, face_group_names = ['active_hull', 'dummy_deck']):
		''' Cut the mesh at z = 0. This is done by the following procedure:
		- Faces which are within the "keep region", z < 0, are marked as unchanged and kept.
		- Faces with one vertex outside the "keep region" are termed bc1 faces, while faces with two vertices outside the "keep region" are termed bc2 faces.
		- For all bc1 faces:
			* Make new vertices at the intersection between edges and the cutting plane.
			* Make two new triangular faces in the now four-sided region between the two new vertices and the two old vertices within the "keep region"
			* Mark the edge which lies along the cutting plane as a "chopping plane edge" originating from a bc1 face. This will later be used when closing the open edge along the cutting plane
			* Combine the original verts_array with an array of new verts.
			* Combine the array of unchanged faces with an array containing all the newly created faces.
		- For all bc2 faces:
			* Make new vertices at the intersection between edges and the cutting plane.
			* Make one new triangular face in the region between the two new vertices and the old vertex within the "keep region"
			* Mark the edge which lies along the cutting plane as a "chopping plane edge" originating from a bc2 face. This will later be used when closing the open edge along the cutting plane
			* Combine the original verts_array with an array of new verts.
			* Combine the array of unchanged faces with an array containing all the newly created faces.
		- Close the open edge along z = 0 by the following procedure:
			* Make a new vertex at the center of the open edge, defined as the average of the coordinates of all vertices along the edge.
			* Create triangular faces between the "chopping plane edges", originating from both bc1 and bc2 edges, and the newly created center vertex of the chopping plane edge. This closes the open edge.

		This whole method assumes the mesh to be triangulated. A separate method for triangulating meshes with non-triangular faces should be made.


		Explanations of vertices and vectors as used below are given in the remainder of the docstring
		* denotes vertices
		\ and / denote parts of vectors

		BC1 face:

		          *                (Outside region)
			     /  \.
                /    \.
               /(v1)  \(v2)
			  /	       \.
             V          V
	--------*------------*---------- (Cutting plane)
	(new vertex 1)     (new vertex 2)


		*                     *           (Keep region)





		BC2 face:

		*                    *  (Outside region)
	     \.                 /
          \.               /
           \(v2)          /(v1)
		    \.           /
			 V          V
	 ---------*--------*------------ (Cutting plane)
	(new vertex 2)     (new vertex 1)


		          *           (Keep region)

		'''

		if np.amax(self.verts_per_face_vec) > 3:
			raise ValueError('chop_at_z0_new_verts assumes the mesh to be triangulated. Triangulate first.')

		normal_axis 							= 2
		self.bc_1_faces_log 					= np.zeros(self.n_faces) #Border crossing with one vertex outside the "keep region"
		self.bc_1_faces_array 					= np.zeros((self.n_faces, 3)).astype('int')
		self.bc_2_faces_log 					= np.zeros(self.n_faces)#Border crossing with two vertices outside the "keep region"
		self.bc_2_faces_array 					= np.zeros((self.n_faces, 3)).astype('int')
		self.uc_faces_log 						= np.zeros(self.n_faces) #Unchanged faces
		self.uc_faces_array 					= np.zeros((self.n_faces, 3)).astype('int')

		self.unchanged_verts_log 				= np.zeros(self.n_verts)
		for i in range(self.n_verts):
			if self.verts_array[i, normal_axis] < 0:
				self.unchanged_verts_log[i] 		= 1

		self.uc_verts_in_faces_log 				= np.zeros(self.n_faces)#Unchanged faces
		bc_1_counter							= 0
		bc_2_counter							= 0
		uc_counter 								= 0
		for i in range(self.n_faces):
			face_verts 								= self.faces_array[i]
			uc_verts_sum 							= 0
			for j in range(3):
				self.uc_verts_in_faces_log[i]			+= self.unchanged_verts_log[face_verts[j]]

		for i in range(self.n_faces):
			face_verts 								= self.faces_array[i]
			if self.uc_verts_in_faces_log[i] == 3:
				self.uc_faces_log[i] 					= 1
				self.uc_faces_array[uc_counter] 		= face_verts
				uc_counter								+= 1
			elif self.uc_verts_in_faces_log[i]	== 2:
				self.bc_1_faces_log[i] 					= 1
				self.bc_1_faces_array[bc_1_counter] 	= face_verts
				bc_1_counter							+= 1
			elif self.uc_verts_in_faces_log[i]	== 1:
				self.bc_2_faces_log[i] 					= 1
				self.bc_2_faces_array[bc_2_counter] 	= face_verts
				bc_2_counter							+= 1

		self.uc_faces_array 					= self.uc_faces_array[~np.all(self.uc_faces_array == 0, axis = 1)]
		self.bc_1_faces_array 					= self.bc_1_faces_array[~np.all(self.bc_1_faces_array == 0, axis = 1)]
		self.bc_2_faces_array 					= self.bc_2_faces_array[~np.all(self.bc_2_faces_array == 0, axis = 1)]


		self.bc_faces_array 					= np.concatenate((self.bc_1_faces_array, self.bc_2_faces_array), axis = 0)
		self.bc_faces_log 						= self.bc_1_faces_log + self.bc_2_faces_log
		n_bc_1_faces 							= int(np.sum(self.bc_1_faces_log))
		n_bc_2_faces 							= int(np.sum(self.bc_2_faces_log))

		n_new_faces_bc1							= int(2*n_bc_1_faces)
		new_faces_array_bc1 					= np.zeros((n_new_faces_bc1, 3)).astype('int')
		n_bc_1_new_verts 						= int(2 * n_bc_1_faces)
		new_verts_bc1 							= np.zeros((n_bc_1_new_verts, 3))
		chop_plane_edges_bc1					= np.zeros((n_bc_1_faces, 2)).astype('int') #Each row contains start and end vertex indices of the edge along the chopping plane. Start and end order are set so that it can be used directly in the new closing face, making a face which has the normal vector facing in or out of the body equally to the bc1 face
		for i in range(n_bc_1_faces):
			# Create two new triangular faces in the part of the bc_1 face which lies within the "keep region"
			relevant_verts 				= self.bc_1_faces_array[i]

			for j in range(3):
				if self.unchanged_verts_log[relevant_verts[j]] != 1:
					outside_vert_local_ind 			= j

			if outside_vert_local_ind == 0:
				v1_end_vert_local_ind				= 1
				v2_end_vert_local_ind				= 2
			elif outside_vert_local_ind == 1:
				v1_end_vert_local_ind				= 2
				v2_end_vert_local_ind				= 0
			elif outside_vert_local_ind == 2:
				v1_end_vert_local_ind				= 0
				v2_end_vert_local_ind				= 1
			else:
				raise ValueError('Erroneous local index of vertex outside of the "keep region"')

			# Move along the edges of the trigangular face, from the top which lies outside of the "keep region", to the intersection between each edge and the "keep region". Generate new verts at the intersections.
			v1_end_vert_global_ind 							= relevant_verts[v1_end_vert_local_ind]
			v2_end_vert_global_ind 							= relevant_verts[v2_end_vert_local_ind]
			outside_vert_global_ind 						= relevant_verts[outside_vert_local_ind]
			v1 												= self.verts_array[v1_end_vert_global_ind] - self.verts_array[outside_vert_global_ind]
			v2 												= self.verts_array[v2_end_vert_global_ind] - self.verts_array[outside_vert_global_ind]
			rel_dist_along_v1 								= self.verts_array[outside_vert_global_ind][normal_axis] / np.absolute(v1[normal_axis])
			rel_dist_along_v2								= self.verts_array[outside_vert_global_ind][normal_axis] / np.absolute(v2[normal_axis])
			new_vert_1 										= self.verts_array[outside_vert_global_ind] + rel_dist_along_v1 * v1
			new_vert_2										= self.verts_array[outside_vert_global_ind] + rel_dist_along_v2 * v2
			new_vert_1_new_array_index 						= 2 * i
			new_vert_2_new_array_index 						= 2 * i + 1
			new_vert_1_global_ind 							= new_vert_1_new_array_index + np.copy(self.n_verts)
			new_vert_2_global_ind 							= new_vert_2_new_array_index + np.copy(self.n_verts)

			new_verts_bc1[new_vert_1_new_array_index]			= new_vert_1
			new_verts_bc1[new_vert_2_new_array_index]			= new_vert_2

			# Generate two triangular faces in the "bottom of the triangle" of the original face, which lie within the "keep region". These are generated between the two new verts at the intersection and the two "bottom verts of the triangle" which both lie within the "keep region"
			new_face_1_global_verts_references 				= np.array((new_vert_1_global_ind, v1_end_vert_global_ind, new_vert_2_global_ind))#global indexing
			new_face_2_global_verts_references 				= np.array((new_vert_2_global_ind, v1_end_vert_global_ind, v2_end_vert_global_ind))#global indexing
			new_face_1_new_array_index 						= 2 * i
			new_face_2_new_array_index 						= 2 * i + 1
			new_faces_array_bc1[new_face_1_new_array_index] = new_face_1_global_verts_references
			new_faces_array_bc1[new_face_2_new_array_index] = new_face_2_global_verts_references

			chop_plane_edges_bc1[i]							= np.array((new_vert_1_global_ind, new_vert_2_global_ind))

		# Combine old and new vertex arrays, now adding bc1 vertices
		self.verts_array				= np.concatenate((self.verts_array, new_verts_bc1), axis = 0)
		self.n_verts 					= self.n_verts + n_bc_1_new_verts

		# Combine old and new face arrays, now adding new faces created to replace bc1 faces
		self.faces_array 				= np.concatenate((self.uc_faces_array, new_faces_array_bc1), axis = 0)
		self.n_faces 					= np.shape(self.faces_array)[0]




		# Create new verts on the cut plane intersecting edges of bc_2 faces. These faces have to vertices outside of the "keep region" and one inside.
		n_bc_2_new_verts				= 2 * n_bc_2_faces
		new_verts_bc2 					= np.zeros((n_bc_2_new_verts, 3))
		n_new_faces_bc2					= n_bc_2_faces
		new_faces_array_bc2				= np.zeros((n_new_faces_bc2, 3)).astype('int')
		chop_plane_edges_bc2			= np.zeros((n_bc_2_faces, 2)).astype('int') #Each row contains start and end vertex indices of the edge along the chopping plane. Start and end order are set so that it can be used directly in the new closing face, making a face which has the normal vector facing in or out of the body equally to the bc2 face
		for i in range(n_bc_2_faces):
			relevant_verts 									= self.bc_2_faces_array[i]

			for j in range(3):
				if self.unchanged_verts_log[relevant_verts[j]] == 1:
					inside_vert_local_ind 						= j

			if inside_vert_local_ind == 0:
				v1_start_vert_local_ind 						= 1
				v2_start_vert_local_ind 						= 2
			elif inside_vert_local_ind == 1:
				v1_start_vert_local_ind 						= 2
				v2_start_vert_local_ind 						= 0
			elif inside_vert_local_ind == 2:
				v1_start_vert_local_ind 						= 0
				v2_start_vert_local_ind 						= 1
			else:
				raise ValueError('Erroneous local index of vertex outside of the "keep region"')

			v1_start_vert_global_ind 						= relevant_verts[v1_start_vert_local_ind]
			v2_start_vert_global_ind 						= relevant_verts[v2_start_vert_local_ind]
			inside_vert_global_ind 							= relevant_verts[inside_vert_local_ind]
			v1												= self.verts_array[inside_vert_global_ind] - self.verts_array[v1_start_vert_global_ind]
			v2												= self.verts_array[inside_vert_global_ind] - self.verts_array[v2_start_vert_global_ind]
			rel_dist_along_v1								= self.verts_array[v1_start_vert_global_ind][normal_axis] / np.absolute(v1[normal_axis])
			rel_dist_along_v2								= self.verts_array[v2_start_vert_global_ind][normal_axis] / np.absolute(v2[normal_axis])
			new_vert_1 										= self.verts_array[v1_start_vert_global_ind] + rel_dist_along_v1 * v1
			new_vert_2 										= self.verts_array[v2_start_vert_global_ind] + rel_dist_along_v2 * v2
			new_vert_1_new_array_index 						= 2 * i
			new_vert_2_new_array_index 						= 2 * i + 1
			new_vert_1_global_ind							= new_vert_1_new_array_index + np.copy(self.n_verts)
			new_vert_2_global_ind							= new_vert_2_new_array_index + np.copy(self.n_verts)

			new_verts_bc2[new_vert_1_new_array_index]		= new_vert_1
			new_verts_bc2[new_vert_2_new_array_index]		= new_vert_2

			new_face_global_verts_references 				= np.array((inside_vert_global_ind, new_vert_1_global_ind, new_vert_2_global_ind))
			new_face_new_array_index 						= i
			new_faces_array_bc2[new_face_new_array_index] 	= new_face_global_verts_references

			chop_plane_edges_bc2[i] 						= np.array((new_vert_2_global_ind, new_vert_1_global_ind))

		# Combine old and new vertex arrays, now adding bc2 vertices
		self.verts_array 					= np.concatenate((self.verts_array, new_verts_bc2), axis = 0)
		self.n_verts 						= self.n_verts + n_bc_2_new_verts

		# Combine old and new face arrays, now adding new faces created to replace bc2 faces
		self.faces_array 					= np.concatenate((self.faces_array, new_faces_array_bc2), axis = 0)
		self.n_faces 						= np.shape(self.faces_array)[0]



		if close_top:
			# Close the now open chop plane edge by creating a new point in its center and thereafter creating triangular faces between each edge and the new point
			new_verts_at_chop_plane 			= np.concatenate((new_verts_bc1, new_verts_bc2), axis = 0)#New verts along the chop plane.
			if new_verts_at_chop_plane.size != 0:
				chop_plane_center_coord				= np.atleast_2d(np.mean(new_verts_at_chop_plane, axis = 0))

				self.verts_array					= np.concatenate((self.verts_array, chop_plane_center_coord), axis = 0)
				self.n_verts 						+= 1
				chop_plane_center_global_ind 		= np.copy(self.n_verts-1)

				## New faces from edges originating from chopped bc1 faces
				n_new_faces_from_bc1_edges			= np.shape(chop_plane_edges_bc1)[0]
				new_faces_from_bc1_edges 			= np.zeros((n_new_faces_from_bc1_edges, 3)).astype('int')
				for i in range(n_new_faces_from_bc1_edges):
					new_faces_from_bc1_edges[i, 0:2] 	= chop_plane_edges_bc1[i]
					new_faces_from_bc1_edges[i, 2] 		= chop_plane_center_global_ind
				self.faces_array 					= np.concatenate((self.faces_array, new_faces_from_bc1_edges), axis = 0)
				self.n_faces 						= np.shape(self.faces_array)[0]

				## New faces from edges originating from chopped bc2 faces
				n_new_faces_from_bc2_edges 			= np.shape(chop_plane_edges_bc2)[0]
				new_faces_from_bc2_edges 			= np.zeros((n_new_faces_from_bc2_edges, 3)).astype('int')
				for i in range(n_new_faces_from_bc2_edges):
					new_faces_from_bc2_edges[i, 0:2] 	= chop_plane_edges_bc2[i]
					new_faces_from_bc2_edges[i, 2] 		= chop_plane_center_global_ind
				self.faces_array 					= np.concatenate((self.faces_array, new_faces_from_bc2_edges), axis = 0)
				self.n_faces 						= np.shape(self.faces_array)[0]

				if split_to_face_groups:
					self.multiple_face_groups				= True
					self.face_group_inds					= np.zeros(self.n_faces)

					n_new_faces 							= n_new_faces_from_bc1_edges + n_new_faces_from_bc2_edges
					for i in range(-n_new_faces, 0):
						self.face_group_inds[i] 				= 1
					#self.face_group_faces_arrays			= []
					#self.face_group_faces_arrays.append(np.copy(self.faces_array[0:-n_new_faces, :]))
					#self.face_group_faces_arrays.append(np.copy(self.faces_array[-n_new_faces:, :]))
					#self.face_group_verts_per_face_vecs 	= []
					#self.face_group_verts_per_face_vecs.append(np.copy(self.verts_per_face_vec[0:-n_new_faces]))
					#self.face_group_verts_per_face_vecs.append(np.copy(self.verts_per_face_vec[-n_new_faces:]))

					self.face_group_names 					= []
					self.face_group_names.append(face_group_names[0])
					self.face_group_names.append(face_group_names[1])

		self.verts_per_face_vec 		= np.zeros(self.n_faces).astype('int')
		for m in range(self.n_faces):
			self.verts_per_face_vec[m] 		= len(self.faces_array[m])



	def rotate(self, rotation_vec):
		rot_array_1				= Rzyx(rotation_vec[0], rotation_vec[1], rotation_vec[2])
		new_verts_array 		= np.zeros((np.shape(self.verts_array)))
		for i in range(self.n_verts):
			new_verts_array[i] 		= np.dot(rot_array_1, self.verts_array[i])
		self.verts_array 		= np.copy(new_verts_array)


	def chop_to_strip(self, x_aft, x_fwd, close_ends = True):
		move_vec_1 				= np.array((-x_aft, 0, 0))
		self.translate(move_vec_1)
		rot_vec_1 				= np.array((0, 0.5 * np.pi, 0))
		self.rotate(rot_vec_1)
		self.chop_at_z0(close_top = close_ends)
		self.rotate(-rot_vec_1)
		self.translate(-move_vec_1)

		move_vec_2 				= np.array((-x_fwd, 0, 0))
		self.translate(move_vec_2)
		rot_vec_2 				= np.array((0, -0.5 * np.pi, 0))
		self.rotate(rot_vec_2)
		self.chop_at_z0(close_top = close_ends)
		self.rotate(-rot_vec_2)
		self.translate(-move_vec_2)




	""" def remove_zero_area_faces(self):
		'''Using jit compiled function. Not yet working'''
		print('Removing zero area faces')
		print('Starting by calculating face properties')
		self.calc_face_properties()
		print('Finished calculating face properties. Starting removal of zero area faces')
		self.faces_array, self.face_areas, self.face_group_inds, self.verts_per_face_vec, self.n_faces = remove_zero_area_faces_jit(self.faces_array.astype('int'), self.face_areas, self.face_group_inds.astype('int'), self.verts_per_face_vec.astype('int'), int(self.n_faces), self.multiple_face_groups)
		print('Finished removing zero area faces') """


	def remove_zero_area_faces(self):
		print('Removing zero area faces')
		print('Original number of faces: 	{:d}'.format(self.n_faces))
		print('Starting by calculating face properties')
		self.calc_face_properties()
		#	print('Finished calculating face properties. Starting removal of zero area faces')
		n_zero_area_faces 						= np.count_nonzero(self.face_areas == 0)
		n_faces_new 							= self.n_faces - n_zero_area_faces
		print('Removing {:d} zero area faces'.format(n_zero_area_faces))
		new_faces 								= np.zeros((n_faces_new, int(np.amax(self.verts_per_face_vec))))
		new_verts_per_face_vec 					= np.zeros(n_faces_new)
		new_face_group_inds 					= np.zeros(n_faces_new)
		counter_new_faces 						= 0
		for i in range(self.n_faces):
			if self.face_areas[i] 					!= 0:
				n_verts_current_face 										= int(self.verts_per_face_vec[i]) #len(self.faces_array[i])
				new_faces[counter_new_faces, 0:n_verts_current_face] 		= self.faces_array[i]
				new_verts_per_face_vec[counter_new_faces] 					= n_verts_current_face
				if self.multiple_face_groups:
					new_face_group_inds[counter_new_faces] 						= self.face_group_inds[i]
				counter_new_faces											+= 1
		self.faces_array 						= new_faces.astype('int')
		self.verts_per_face_vec 				= new_verts_per_face_vec.astype('int')
		if self.multiple_face_groups:
			self.face_group_inds 					= new_face_group_inds.astype('int')
		self.n_faces 							= n_faces_new
		print('Finished removing zero area faces')
		print('New number of faces: 		{:d}'.format(self.n_faces))

	"""def remove_unused_verts(self):
		'''Old non-jit version'''
		print('Removing unused vertices')
		do_not_remove_verts_decision_array  	= np.zeros(self.n_verts)
		for i in range(self.n_verts):
			if i in self.faces_array:
				do_not_remove_verts_decision_array[i] 	= 1

		
		n_used_verts 							= int(np.sum(do_not_remove_verts_decision_array))
		new_verts_array							= np.zeros((n_used_verts, 3))
		new_verts_array_counter 				= 0
		for i in range(self.n_verts):
			if do_not_remove_verts_decision_array[i] == 1:
				new_verts_array[new_verts_array_counter] 	= self.verts_array[i]
				self.faces_array[self.faces_array == i]		= new_verts_array_counter #Renumber vertex references in faces
				new_verts_array_counter 					+= 1
		self.verts_array 						= new_verts_array
		self.n_verts 							= n_used_verts
		print('Finished removing unused vertices')"""

	def remove_unused_verts(self):
		print('Removing unused vertices')
		print('Identifying unused vertices.')
		do_not_remove_verts_decision_array 		= identify_unused_verts_jit(self.faces_array, self.verts_array, self.n_verts)
		n_used_verts 							= int(np.sum(do_not_remove_verts_decision_array))
		n_unused_verts  						= self.n_verts - n_used_verts
		print('Identified {:d} unused vertices'.format(n_unused_verts))

		if n_unused_verts > 0:
			faces_array, verts_array, n_verts 		= remove_unused_verts_from_arrays_jit(self.faces_array, self.verts_array, self.n_verts, n_used_verts, do_not_remove_verts_decision_array)
			self.faces_array  						= faces_array
			self.verts_array 						= verts_array
			self.n_verts 							= n_verts
		print('Finished removing unused vertices')

	def triangulate(self):
		''' Triangulate faces which orinally contain more than 3 vertices '''
		print('n_faces before triangulation:   {:d}'.format(self.n_faces))
		faces_array_keep 				= np.zeros((np.shape(self.faces_array)[0], 3)).astype('int')
		faces_array_additions 			= np.zeros((10*np.shape(self.faces_array)[0], 3)).astype('int')
		faces_array_additions_counter 	= 0
		for i in range(self.n_faces):
			if self.verts_per_face_vec[i] == 3:
				faces_array_keep[i] 	= self.faces_array[i, 0:3]
			else:
				#print(i)
				n_verts_org 		= self.verts_per_face_vec[i]
				n_new_faces_current = n_verts_org - 2 #If four verts: two triangulated faces.
				verts_org 			= self.faces_array[i]
				for j in range(n_new_faces_current):
					vert_1_local_ind			= 0
					vert_2_local_ind 			= 1 + j
					vert_3_local_ind 			= 2 + j
					vert_1_global_ind 			= verts_org[vert_1_local_ind]
					vert_2_global_ind 			= verts_org[vert_2_local_ind]
					vert_3_global_ind 			= verts_org[vert_3_local_ind]
					new_face 					= np.array((vert_1_global_ind, vert_2_global_ind, vert_3_global_ind))
					faces_array_additions[faces_array_additions_counter] = new_face
					faces_array_additions_counter 	+= 1


		faces_array_keep 		= faces_array_keep[~np.all(faces_array_keep == 0, axis = 1)]
		faces_array_additions 	= faces_array_additions[~np.all(faces_array_additions == 0, axis = 1)]
		self.faces_array 		= np.concatenate((faces_array_keep, faces_array_additions), axis = 0)
		self.n_faces			= np.shape(self.faces_array)[0]
		self.verts_per_face_vec	= np.zeros(self.n_faces).astype('int')
		for i in range(self.n_faces):
			self.verts_per_face_vec[i] 	= len(self.faces_array[i, :])
		self.n_verts 			= np.shape(self.verts_array)[0]
		print('n_faces after triangulation:   {:d}'.format(self.n_faces))

	def combine_equal_verts(self):
		''' Combine repeating verts. Verts are also be sorted by coordinates. '''

		verts_array_new, indx, inverse, count 		= np.unique(self.verts_array, return_index = True, return_inverse = True, return_counts=True, axis = 0)
		# 'inverse' has equal length to the original array. Each entry contains the new index of the entry that used to be in that entry's position.
		print(inverse)
		faces_array_new 						= np.zeros(np.shape(self.faces_array))
		for i in range(self.n_verts):
			new_index_of_this_entry 						= inverse[i]
			faces_array_new[self.faces_array == i] 		= new_index_of_this_entry
		self.faces_array 							= faces_array_new.astype('int')
		n_verts_new 								= len(indx)
		self.verts_array 							= verts_array_new
		self.n_verts 								= n_verts_new
		print(verts_array_new)

	#---------------------------------------------------------------------------------------
	# Read and write routines
	#---------------------------------------------------------------------------------------
	def import_from_obj(self, file_path):
		file 						= open(file_path, 'r')
		lines 						= file.readlines()
		self.n_lines				= len(lines)
		self.n_verts 				= 0
		self.n_faces 				= 0
		start_line_verts 			= False
		start_line_faces 			= False

		self.face_group_names 					= []
		n_faces_per_face_group                  = []


		n_groups_counted 						= 0
		s_lines_since_group_addition 			= 0
		for i in range(self.n_lines):
			line 						= lines[i]
			if line[0:2] == 'v ':
				if start_line_verts == False:
					start_line_verts 	= i
				self.n_verts 		+= 1
			elif line[0:2] == 'f ':
				if start_line_faces == False:
					start_line_faces 	= i
				self.n_faces 			+= 1
			elif line[0:2] == 'g ':
				self.multiple_face_groups 	= True
				self.face_group_names.append(line[2:-1])
				if n_groups_counted >= 1:
					n_lines_in_previous_group 	= i - prev_group_start_line - 1 - s_lines_since_group_addition
					n_faces_per_face_group.append(n_lines_in_previous_group)
				prev_group_start_line 		= i
				n_groups_counted 			+= 1
				s_lines_since_group_addition= 0
			elif line[0:2] == 's ':
				s_lines_since_group_addition 	+= 1
			if i == self.n_lines - 1 and self.multiple_face_groups == True:
				n_lines_in_previous_group 	= i - prev_group_start_line
				n_faces_per_face_group.append(n_lines_in_previous_group)

		n_face_groups 							= len(self.face_group_names)

		self.verts_per_face_vec 	= np.zeros(self.n_faces).astype('int')
		face_counter 				= 0
		for i in range(self.n_faces + n_face_groups - 1):
			line_no 						= start_line_faces + i
			text_line 						= lines[line_no]
			if text_line[0:2] != 'g ':
				splitted_line 							= text_line.strip('').split(' ')
				if splitted_line[-1] == '\n':
					n_verts_current								= len(splitted_line) - 2 #counts both the initial f and the ending \n
				else:
					n_verts_current 							= len(splitted_line) - 1 #counts the initial f
				self.verts_per_face_vec[face_counter] 	= n_verts_current
				face_counter 							+= 1

		self.verts_array 			= np.zeros((self.n_verts, 3))
		self.faces_array 			= -1 * np.ones((self.n_faces, int(np.amax(self.verts_per_face_vec))))

		for i in range(self.n_verts):
			line_no 						= start_line_verts + i
			splitted_line 					= lines[line_no].strip('').split(' ')
			for j in range(3):
				column_no 						= j + 1
				self.verts_array[i, j] 			= np.float(splitted_line[column_no])

		face_counter = 0
		for i in range(self.n_faces + n_face_groups - 1):
			line_no 						= start_line_faces + i
			splitted_line 					= lines[line_no].strip('').split(' ')
			if lines[line_no][0:2] != 'g ':
				for j in range(int(self.verts_per_face_vec[face_counter])):
					column_no 						= j + 1
					column_content 					= splitted_line[column_no]
					if '/' in column_content:
						first_slash_ind 				= column_content.find('/')
						column_content 					= column_content[0:first_slash_ind]#Remove texture and/or vertex normal data
					self.faces_array[face_counter, j] 	= np.int(column_content) - 1 #-1 due to 1 indexing in wavefront .obj file
				face_counter 	+= 1

		self.faces_array 					= self.faces_array.astype('int')


		self.face_group_faces_arrays			= []
		self.face_group_verts_per_face_vecs		= []
		self.face_group_inds					= np.zeros(self.n_faces)
		end_ind_current 			 			= 0
		for i in range(n_face_groups):
			n_faces_in_current 						= n_faces_per_face_group[i]
			start_ind_current 						= end_ind_current
			end_ind_current 						= start_ind_current + n_faces_in_current
			faces_array_current 					= self.faces_array[start_ind_current:end_ind_current]
			self.face_group_faces_arrays.append(faces_array_current)
			verts_per_face_vec_current 				= self.verts_per_face_vec[start_ind_current:end_ind_current]
			self.face_group_verts_per_face_vecs.append(verts_per_face_vec_current)
			self.face_group_inds[start_ind_current:end_ind_current] 	= i * np.ones(end_ind_current - start_ind_current)

	def import_from_verts_and_faces_arrays(self, verts_array, faces_array):
		''' Import verts_array and faces_array directly.  Faces_array must be zero indexed, i.e. follow
		the conventions in my_mesh and not the convention of the Wavefront file format. '''
		self.verts_array 						= verts_array
		self.faces_array 						= faces_array.astype('int')

		self.n_verts 							= np.shape(self.verts_array)[0]
		self.n_faces 							= np.shape(self.faces_array)[0]

		self.verts_per_face_vec 				= np.zeros(self.n_faces).astype('int')
		for i in range(self.n_faces):
			self.verts_per_face_vec[i] 				= int(len(self.faces_array[i, :]))

		self.multiple_face_groups 				= True

		self.face_group_faces_arrays			= []
		self.face_group_verts_per_face_vecs		= []
		self.face_group_inds					= np.zeros(self.n_faces)
		self.face_group_faces_arrays.append(self.faces_array)
		self.face_group_verts_per_face_vecs.append(self.verts_per_face_vec)

	def write_obj(self, output_file_path):
		outfile 							= open(output_file_path, 'w')
		outfile.write('# Mesh file exported from the my_mesh program by the Department of Marine Technology, NTNU\n')
		outfile.write('#\n')
		outfile.write('o object\n')
		for i in range(self.n_verts):
			outfile.write('v {:.9f} {:.9f} {:.9f}\n'.format(self.verts_array[i, 0], self.verts_array[i, 1], self.verts_array[i, 2]))
		if self.multiple_face_groups:
			n_face_groups 			= len(self.face_group_names)
			self.face_group_faces_arrays			= []
			self.face_group_verts_per_face_vecs 	= []
			for i in range(n_face_groups):
				faces_inds 					= np.where(self.face_group_inds == i)[0]
				self.face_group_faces_arrays.append(np.copy(self.faces_array[faces_inds, :]))
				self.face_group_verts_per_face_vecs.append(np.copy(self.verts_per_face_vec[faces_inds]))
			for m in range(n_face_groups):
				outfile.write('g ' + self.face_group_names[m] + '\n')
				current_faces_array 		= self.face_group_faces_arrays[m]
				current_n_faces				= np.shape(current_faces_array)[0]
				current_verts_per_faces_vec	= self.face_group_verts_per_face_vecs[m]
				for i in range(current_n_faces):
					outfile.write('f ')
					n_verts_current_face		= int(current_verts_per_faces_vec[i])
					for j in range(n_verts_current_face):
						outfile.write('{:d} '.format(int(current_faces_array[i, j]) + 1)) #+1 due to 1 indexing in wavefront .obj file
					outfile.write('\n')
		else:
			outfile.write('g object_group\n')
			for i in range(self.n_faces):
				outfile.write('f ')
				n_verts_current_face 	= int(self.verts_per_face_vec[i])
				for j in range(n_verts_current_face):
					outfile.write('{:d} '.format(int(self.faces_array[i, j]) + 1)) #+1 due to 1 indexing in wavefront .obj file
				outfile.write('\n')
		outfile.close()

	#---------------------------------------------------------------------------------------
	# Create 3D meshes from 2D section data
	#---------------------------------------------------------------------------------------

	def extrude_from_2d_data(self, input_verts_2d, extrusion_length, two_way_extrusion = True, close_planar_ends = False, foil_group_names = False, wing_group_names = False):
		''' Make a 3D mesh from 2D profile data by extruding out of the original plane.
		Assumes the original plane to be an xy plane and extrudes in the positive z direction.
		input_verts_2d should be an N x 2 numpy array, where N is the number of points. The first column contains x coordinates and the second y coordinates
		The method will close the extruded mesh along any open end between the first and the last 2d vertex. If one also wants it to close the planar ends then close_planar_ends should be set to true. NB: This might cause
		trouble with meshes where one cannot create straight lines between all vertices and the center point of the plane, e.g. for thin foil sections with large flap angles.'''
		if two_way_extrusion:
			extrusion_length					*= 2
		n_2d_verts 							= np.shape(input_verts_2d)[0]
		self.n_verts 						= 2 * n_2d_verts
		self.verts_array 					= np.zeros((self.n_verts, 3))
		self.verts_array[:n_2d_verts, 0:2]  = input_verts_2d
		self.verts_array[n_2d_verts:, 0:2]  = input_verts_2d
		self.verts_array[n_2d_verts:, 2]	= extrusion_length * np.ones(n_2d_verts)

		if input_verts_2d[0, 0] != input_verts_2d[-1, 0] or input_verts_2d[0, 1] != input_verts_2d[-1, 1]:
			close_open_end 						= True
		else:
			close_open_end						= False
		n_face_pairs 						= n_2d_verts - 1
		if close_open_end:
			n_face_pairs 						+= 1

		self.n_faces 						= 2 * n_face_pairs
		self.faces_array 					= np.zeros((self.n_faces, 3)).astype('int')
		self.verts_per_face_vec				= 3 * np.ones(self.n_faces).astype('int')
		start_ind_second_plane 				= n_2d_verts
		for i in range(n_2d_verts - 1):
			first_face_index 						= 2 * i
			second_face_index 						= 2 * i + 1
			self.faces_array[first_face_index, 0] 	= i
			self.faces_array[first_face_index, 1] 	= start_ind_second_plane + i
			self.faces_array[first_face_index, 2] 	= i + 1

			self.faces_array[second_face_index, 0] 	= i + 1
			self.faces_array[second_face_index, 1] 	= start_ind_second_plane + i
			self.faces_array[second_face_index, 2] 	= start_ind_second_plane + i + 1

		if close_open_end:
			first_face_index 						= 2 * (n_2d_verts - 1)
			second_face_index 						= 2 * (n_2d_verts - 1) + 1
			end_ind_first_plane 					= n_2d_verts - 1
			self.faces_array[first_face_index, 0]	= end_ind_first_plane
			self.faces_array[first_face_index, 1]	= self.n_verts - 1
			self.faces_array[first_face_index, 2]	= 0

			self.faces_array[second_face_index, 0]	= 0
			self.faces_array[second_face_index, 1]	= self.n_verts - 1
			self.faces_array[second_face_index, 2]	= start_ind_second_plane
		if foil_group_names and wing_group_names:
			raise ValueError('Cannot name the output both as foil group names and wing group names')
		if foil_group_names or wing_group_names:
			self.multiple_face_groups				= True
			# Create vector of face indices and set trailing edge indices (the last two faces created) to a separate value
			self.face_group_inds					= np.zeros(self.n_faces)
			self.face_group_inds[-2:] 				= np.ones(2)
		if foil_group_names:
			# Create names of face groups
			self.face_group_names 					= []
			self.face_group_names.append('foil_foil_foil')
			self.face_group_names.append('foil_foil_trailingEdge')
		elif wing_group_names:
			# Create names of face groups
			self.face_group_names 					= []
			self.face_group_names.append('wing_wing_wing')
			self.face_group_names.append('wing_wing_trailingEdge')

		if close_planar_ends:
			# Close planar ends in the plane of imported points and in the end plane of the extrusion
			# Start by adding a point on the geometrical center of each plane, then create triangular faces between all other in-plane verts and the new center vert.
			new_vert_1 									= np.zeros(3)
			new_vert_1[0] 								= np.average(self.verts_array[0:n_2d_verts, 0])
			new_vert_1[1] 								= np.average(self.verts_array[0:n_2d_verts, 1])
			new_vert_2 									= np.copy(new_vert_1)
			new_vert_2[2] 								= extrusion_length
			new_verts_array 							= np.zeros((2, 3))
			new_verts_array[0, :]						= new_vert_1
			new_verts_array[1, :]						= new_vert_2
			self.verts_array 							= np.concatenate((self.verts_array, new_verts_array), axis = 0)
			new_vert_1_index 							= self.n_verts
			new_vert_2_index 							= self.n_verts + 1
			self.n_verts 								+= 2
			if close_open_end:
				n_new_faces_one_plane 						= n_2d_verts
			else:
				n_new_faces_one_plane 						= n_2d_verts - 1

			new_faces_array_1 							= np.zeros((n_new_faces_one_plane, 3)).astype('int')
			for i in range(n_2d_verts - 1):
				new_faces_array_1[i, 0]						= i + 1
				new_faces_array_1[i, 1]						= new_vert_1_index
				new_faces_array_1[i, 2]						= i
			if close_open_end:
				new_faces_array_1[n_2d_verts - 1, 0]		= 0
				new_faces_array_1[n_2d_verts - 1, 1]		= new_vert_1_index
				new_faces_array_1[n_2d_verts - 1, 2]		= end_ind_first_plane

			new_faces_array_2 							= np.copy(new_faces_array_1)
			new_faces_array_2[:, 0]						+= n_2d_verts
			new_faces_array_2[:, 2]						+= n_2d_verts
			new_faces_array_2[:, 1]						= new_vert_2_index

			self.faces_array 							= np.concatenate((self.faces_array, new_faces_array_1, new_faces_array_2), axis = 0)
			self.n_faces 								= np.shape(self.faces_array)[0]
			self.verts_per_face_vec						= 3 * np.ones(self.n_faces).astype('int')

			if foil_group_names:
				new_face_group_inds 						= 2 * np.ones(2 * n_new_faces_one_plane)
				self.face_group_inds 						= np.concatenate((self.face_group_inds, new_face_group_inds), axis = 0)
				self.face_group_names.append('foil_foil_ends')
			elif wing_group_names:
				new_face_group_inds 						= 2 * np.ones(2 * n_new_faces_one_plane)
				self.face_group_inds 						= np.concatenate((self.face_group_inds, new_face_group_inds), axis = 0)
				self.face_group_names.append('wing_wing_ends')

		if two_way_extrusion:
			translation_vec 							= np.array((0, 0, -0.5*extrusion_length))
			self.translate(translation_vec)

	def combine_section_data(self, sections_array, surface_name = 'geometry_surface', separate_closing_surface_name = None, close_ends = False, ends_surface_name = None, separate_downstream_faces_name = None, length_ratio_downstream_verts = 0):
		''' Combine section point data to a 3D mesh.
		sections_array is (n_sections, n_points_per_section, 3) '''
		n_sections 										= np.shape(sections_array)[0]
		n_points_per_section							= np.shape(sections_array)[1]

		self.n_verts 									= n_sections * n_points_per_section
		self.n_faces 									= (n_sections - 1) * (2 * n_points_per_section)
		self.verts_array								= np.zeros((self.n_verts, 3))
		self.faces_array 								= np.zeros((self.n_faces, 3))
		self.verts_per_face_vec							= 3 * np.ones(self.n_faces)

		# Add all verts to self.verts_array
		for i in range(n_sections):
			start_ind_verts_current_s						= i * n_points_per_section
			start_ind_verts_next_s 							= start_ind_verts_current_s + n_points_per_section
			self.verts_array[start_ind_verts_current_s:start_ind_verts_next_s]	 = sections_array[i, :, :]
		# Create between-section triangular faces and add to self.faces_array
		faces_counter 									= 0
		for i in range(n_sections - 1):
			start_ind_verts_current_s						= i * n_points_per_section
			start_ind_verts_next_s 							= start_ind_verts_current_s + n_points_per_section
			for j in range(n_points_per_section - 1):
				start_sect_start_vert_ind 						= start_ind_verts_current_s + j
				start_sect_end_vert_ind							= start_ind_verts_current_s + j + 1
				end_sect_start_vert_ind 						= start_ind_verts_next_s + j
				end_sect_end_vert_ind							= start_ind_verts_next_s + j + 1

				face_1 											= np.array((start_sect_start_vert_ind, start_sect_end_vert_ind, end_sect_start_vert_ind))
				face_2 											= np.array((start_sect_end_vert_ind, end_sect_end_vert_ind, end_sect_start_vert_ind))

				self.faces_array[faces_counter] 				= face_1
				faces_counter 									+= 1
				self.faces_array[faces_counter] 				= face_2
				faces_counter 									+= 1
			# Closing surface between first and last points at each section.
			end_ind_verts_current_s 					= start_ind_verts_next_s - 1
			end_ind_verts_next_s 						= start_ind_verts_next_s + n_points_per_section - 1

			self.faces_array[faces_counter] 			= np.array((end_ind_verts_current_s, end_ind_verts_next_s, start_ind_verts_next_s))
			faces_counter 								+= 1
			self.faces_array[faces_counter] 			= np.array((start_ind_verts_current_s, end_ind_verts_current_s, start_ind_verts_next_s))
			faces_counter 								+= 1



		# Set face group indices and face group names
		if surface_name != None:
			self.multiple_face_groups						= True
			self.face_group_inds 							= np.zeros(self.n_faces)
			self.face_group_names 							= []
			self.face_group_names.append(surface_name)
		if separate_closing_surface_name != None:
			# Add a separate name for the closing surface faces, created in the region between the start and end points for each section. Only to be used if these do not coincide, e.g. with an open trailing edge of a foil section
			self.face_group_names.append(separate_closing_surface_name)
			for i in range(n_sections - 1):
				start_face_face_index 			= i * 2 * n_points_per_section
				start_face_face_index_next_sect	= (i + 1) * 2 * n_points_per_section
				self.face_group_inds[start_face_face_index_next_sect - 2:start_face_face_index_next_sect]	= np.ones(2)
		
		if separate_downstream_faces_name != None:
			if separate_closing_surface_name != None:
				# Add a separate name for the most downstream faces, including potentially the closing surface.
				self.face_group_names.append(separate_downstream_faces_name)
				max_face_group_ind_so_far 								= np.amax(self.face_group_inds)
				for i in range(n_sections - 1):
					start_face_face_index 									= i * 2 * n_points_per_section
					start_face_face_index_next_sect							= (i + 1) * 2 * n_points_per_section
					faces_subarray_current 									= self.faces_array[start_face_face_index:start_face_face_index_next_sect].astype(int)
					
					current_verts 											= self.verts_array[faces_subarray_current.flatten()]
					upstream_end 											= np.amin(current_verts[:, 0])
					downstream_end 											= np.amax(current_verts[:, 0])
					current_chord 											= downstream_end - upstream_end
					x_lim_downstream_faces 									= upstream_end + (1 - length_ratio_downstream_verts) * current_chord
					n_faces_current 										= np.shape(faces_subarray_current)[0]
					## Identify global indices of faces with vertices placed downstream of x_lim_downstream_faces (i.e. vertices that have the first coordinate greater than x_lim_downstream_faces)
					downstream_faces_global_indices 						= []
					for j in range(n_faces_current):
						if np.amax(self.verts_array[faces_subarray_current[j]][:, 0]) >= x_lim_downstream_faces:
							downstream_faces_global_indices.append(start_face_face_index + j)
					self.face_group_inds[downstream_faces_global_indices]	= (max_face_group_ind_so_far + 1) * np.ones(len(downstream_faces_global_indices))

				# Add a separate name for the closing surface faces, created in the region between the start and end points for each section. Only to be used if these do not coincide, e.g. with an open trailing edge of a foil section
				self.face_group_names.append(separate_closing_surface_name)
				max_face_group_ind_so_far 								= np.amax(self.face_group_inds)
				for i in range(n_sections - 1):
					start_face_face_index 			= i * 2 * n_points_per_section
					start_face_face_index_next_sect	= (i + 1) * 2 * n_points_per_section
					self.face_group_inds[start_face_face_index_next_sect - 2:start_face_face_index_next_sect]	= (max_face_group_ind_so_far + 1) * np.ones(2)
			else:
				# Add a separate name for the most downstream faces, including potentially the closing surface.
				self.face_group_names.append(separate_downstream_faces_name)
				for i in range(n_sections - 1):
					start_face_face_index 									= i * 2 * n_points_per_section
					start_face_face_index_next_sect							= (i + 1) * 2 * n_points_per_section
					faces_subarray_current 									= self.faces_array[start_face_face_index:start_face_face_index_next_sect].astype(int)
					
					current_verts 											= self.verts_array[faces_subarray_current.flatten()]
					upstream_end 											= np.amin(current_verts[:, 0])
					downstream_end 											= np.amax(current_verts[:, 0])
					current_chord 											= downstream_end - upstream_end
					x_lim_downstream_faces 									= upstream_end + (1 - length_ratio_downstream_verts) * current_chord
					n_faces_current 										= np.shape(faces_subarray_current)[0]
					## Identify global indices of faces with vertices placed downstream of x_lim_downstream_faces (i.e. vertices that have the first coordinate greater than x_lim_downstream_faces)
					downstream_faces_global_indices 						= []
					for j in range(n_faces_current):
						if np.amax(self.verts_array[faces_subarray_current[j]][:, 0]) >= x_lim_downstream_faces:
							downstream_faces_global_indices.append(start_face_face_index + j)
					self.face_group_inds[downstream_faces_global_indices]	= np.ones(len(downstream_faces_global_indices))
				

		# Close ends if desired
		if close_ends:
			ends_faces_new_verts_array 						= np.zeros((2, 3))
			for i in range(3):
				ends_faces_new_verts_array[0, i] 				= np.average(sections_array[0, :, i])
				ends_faces_new_verts_array[1, i] 				= np.average(sections_array[-1, :, i])
			self.verts_array 								= np.concatenate((self.verts_array, ends_faces_new_verts_array), axis = 0)
			self.n_verts 									= np.shape(self.verts_array)[0]

			n_ends_faces 									= 2 * n_points_per_section
			ends_faces_array	 							= np.zeros((n_ends_faces, 3))
			ends_faces_verts_per_face_vec					= 3 * np.ones(n_ends_faces)
			new_point_first_end_ind 						= self.n_verts - 2
			new_point_second_end_ind 						= self.n_verts - 1
			#n_verts_before_end_center_points 				= self.n_verts - 2
			for i in range(n_points_per_section - 1):
				face_ind_first_end 							= i#Index of the face in the ends_faces_array
				start_vert_ind_first_end 					= i
				next_vert_ind_first_end 					= i + 1
				ends_faces_array[face_ind_first_end]		= np.array((start_vert_ind_first_end, new_point_first_end_ind, next_vert_ind_first_end))

				face_ind_second_end 						= i + n_points_per_section #Index of the face in the ends_faces_array
				start_vert_ind_second_end 					= start_vert_ind_first_end + (n_sections - 1) * n_points_per_section
				next_vert_ind_second_end 					= start_vert_ind_second_end + 1
				ends_faces_array[face_ind_second_end]		= np.array((start_vert_ind_second_end, next_vert_ind_second_end, new_point_second_end_ind))

			face_ind_first_end 								+= 1
			first_vert_ind_first_end 						= 0
			last_vert_ind_first_end 						= n_points_per_section - 1
			ends_faces_array[face_ind_first_end]			= np.array((last_vert_ind_first_end, new_point_first_end_ind, first_vert_ind_first_end))

			face_ind_second_end								+= 1
			first_vert_ind_second_end 						= first_vert_ind_first_end + (n_sections - 1) * n_points_per_section
			last_vert_ind_second_end 						= last_vert_ind_first_end + (n_sections - 1) * n_points_per_section
			ends_faces_array[face_ind_second_end]			= np.array((first_vert_ind_second_end, new_point_second_end_ind, last_vert_ind_second_end))

			self.faces_array 								= np.concatenate((self.faces_array, ends_faces_array), axis = 0)
			self.verts_per_face_vec							= np.concatenate((self.verts_per_face_vec, ends_faces_verts_per_face_vec))
			self.n_faces									= np.shape(self.faces_array)[0]
			if ends_surface_name != None:
				self.face_group_names.append(ends_surface_name)
				ends_face_group_inds 							= (1 + np.amax(self.face_group_inds)) * np.ones(n_ends_faces)#2 * np.ones(n_ends_faces)
			else:
				ends_face_group_inds							= np.zeros(n_ends_faces)
			if self.multiple_face_groups:
				self.face_group_inds 							= np.concatenate((self.face_group_inds, ends_face_group_inds))
