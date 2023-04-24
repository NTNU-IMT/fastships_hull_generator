# © 2023, NTNU
# Author: John Martin Kleven Godø <john.martin.godo@ntnu.no>
# This code is licenced under the GNU General Public License v3.0

import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
from datetime import datetime

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import fastships_open.hull_generator.cross_sections as cross_sections
import fastships_open.mesh_tools.mesh as mesh



def cos_transition_func(dimless_par):
	''' Cosine based smooth transition vector from 1 to 0. 1 when dimless_par is 0 and 0 when dimless_par is 1 '''
	return 0.5*(1+np.cos(np.pi*dimless_par))


class Hull():
	'''General hull shape class, for creation of demihulls for slender catamarans.'''
	def __init__(self):
		a = 0
		self.close_mesh						= True #Change to false if a hull mesh without a water-tight transom and deck is desired
		self.verts_generated 				= False
		self.faces_generated 				= False
		self.edgy_cs 						= False
		self.n_flat_sides_subm_edgy_cs 		= 3
		self.n_flat_sides_flare_edgy_cs 	= 2

		self.vessel_type					= 'fast-ferry'#'fast-ferry' or 'wigley'

	def set_resolution(self, n_cs, n_cs_verts_subm_full):
		''' Set numerical resolution.
		n_cs:                                 number of cross-sections
		n_cs_verts_full:                      number of vertices per cross-section, both sides combined '''
		self.n_cs 							= n_cs
		if self.edgy_cs:
			print('n_cs_verts_subm_full will be neglected, as the hull has edgy cross-sections')
			self.n_cs_verts_subm_half 			= self.n_flat_sides_subm_edgy_cs + 1
		else:
			self.n_cs_verts_subm_half  			= int(np.ceil(0.5*n_cs_verts_subm_full))

	def set_main_dimensions(self, l_wl):
		self.l_wl 							= l_wl

	def set_width_data(self, width_stern, entrance_angle_side_to_side, const_width_fwd_rel):
		''' Set data for lingitudinal distribution of width
		width_stern:                          width in waterline at stern
		const_width_fwd_rel: 				  forward end of constant-width part of hull, relative to l_wl'''
		self.width_stern 					= width_stern
		self.entrance_angle_half			= 0.5 * entrance_angle_side_to_side
		self.const_width_fwd_abs 			= self.l_wl*const_width_fwd_rel

	def set_draft_data(self, draft_max, draft_transom, max_draft_aft_rel, max_draft_fwd_rel, draft_dist_el_exp_fwd):
		''' Set data for lingitudinal distribution of draft '''
		self.draft_max 						= draft_max
		self.draft_transom 					= draft_transom
		self.max_draft_aft 					= self.l_wl * max_draft_aft_rel
		self.max_draft_fwd 					= self.l_wl * max_draft_fwd_rel
		self.draft_dist_el_exp_fwd 			= draft_dist_el_exp_fwd

	def set_cs_el_exp_data(self, cs_el_exp_aft, cs_el_exp_fwd, const_cs_el_exp_aft_rel, const_cs_el_exp_fwd_rel):
		''' Set data for longitudinal distribution of cross-sectional superellipse exponent '''
		self.cs_el_exp_aft 					= cs_el_exp_aft
		self.cs_el_exp_fwd 					= cs_el_exp_fwd
		self.const_cs_el_exp_aft 			= self.l_wl * const_cs_el_exp_aft_rel
		self.const_cs_el_exp_fwd 			= self.l_wl * const_cs_el_exp_fwd_rel

	def set_abv_w_data(self, n_cs_verts_abv_w_full, abv_w_flare_h_aft, abv_w_flare_h_fwd, abv_w_flare_w_aft, const_abv_w_flare_fwd_rel, abv_w_flare_dist_se_exp, h_main_deck):
		if self.edgy_cs:
			print('Overriding self.n_cs_verts_abv_w_half to self.n_flat_sides_flare + 1 + 2')
			self.n_cs_verts_abv_w_half 			= self.n_flat_sides_flare_edgy_cs + 1 + 2 # 2 verts on the flat side above the flare
		else:
			self.n_cs_verts_abv_w_half 			= int(np.ceil(0.5*n_cs_verts_abv_w_full))
		self.n_cs_verts_abv_w_flare_half	= self.n_cs_verts_abv_w_half - 2
		self.abv_w_flare_h_aft				= abv_w_flare_h_aft
		self.abv_w_flare_h_fwd				= abv_w_flare_h_fwd
		self.abv_w_flare_w_aft				= abv_w_flare_w_aft
		self.const_abv_w_flare_fwd			= self.l_wl * const_abv_w_flare_fwd_rel
		self.abv_w_flare_dist_se_exp		= abv_w_flare_dist_se_exp
		self.h_main_deck 					= h_main_deck

	def generate_long_property_distributions(self):
		if self.vessel_type == 'wigley':
			self.generate_long_property_distributions_wigley_hull()
		elif self.vessel_type == 'fast-ferry':
			self.generate_long_property_distributions_fastferry()
		else:
			raise ValueError('self.vessel_type not recognized')

	def generate_long_property_distributions_wigley_hull(self):
		''' Treating self.width_stern as the maximum width.  '''
		self.width_vec 						= np.zeros(self.n_cs)
		self.draft_vec 						= np.zeros(self.n_cs)
		self.cs_el_exp_vec 					= 2 * np.ones(self.n_cs)
		self.abv_w_flare_h_vec 				= np.zeros(self.n_cs)
		self.abv_w_flare_w_vec				= np.zeros(self.n_cs)
		self.station_x_vec					= np.linspace(0, self.l_wl, self.n_cs)

		for i in range(self.n_cs):
			station_x 							= self.station_x_vec[i]

			# Width
			if station_x <= self.l_wl:
				dist_from_bow 						= self.l_wl - station_x
				dist_from_stern                     = station_x
				dist_from_extremity 				= np.minimum(dist_from_bow, dist_from_stern)
				dimless_dist_from_extremity 		= dist_from_extremity / (0.5 * self.l_wl)
				dimless_dist_from_midships          = 1 - dimless_dist_from_extremity
				self.width_vec[i] 					= self.width_stern * (1 - dimless_dist_from_midships**2)
			else:
				raise ValueError('Invalid value of station_x occured')

			# Draft
			if station_x <= self.l_wl:
				self.draft_vec[i]					= self.draft_max
			else:
				raise ValueError('Invalid value of station_x occured')

			# Above-water flare
			if station_x <= self.l_wl:
				self.abv_w_flare_w_vec[i] 			= 0
				self.abv_w_flare_h_vec[i] 			= self.draft_max
			else:
				raise ValueError('Invalid value of station_x occured')

	def generate_long_property_distributions_fastferry(self):
		self.width_vec 						= np.zeros(self.n_cs)
		self.draft_vec 						= np.zeros(self.n_cs)
		self.cs_el_exp_vec 					= np.zeros(self.n_cs)
		self.abv_w_flare_h_vec 				= np.zeros(self.n_cs)
		self.abv_w_flare_w_vec				= np.zeros(self.n_cs)
		self.station_x_vec					= np.linspace(0, self.l_wl, self.n_cs)

		for i in range(self.n_cs):
			station_x 							= self.station_x_vec[i]

			# Width
			if station_x  <= self.const_width_fwd_abs and station_x >= 0:
				self.width_vec[i]					= self.width_stern
			elif station_x <= self.l_wl:
				dist_from_bow 						= self.l_wl - station_x
				width_linear_model 					= dist_from_bow * np.tan(self.entrance_angle_half)

				width_full_width 					= self.width_stern

				dimless_pos_transition_width		= (station_x - self.const_width_fwd_abs) / (self.l_wl - self.const_width_fwd_abs)
				full_width_weight 					= cos_transition_func(dimless_pos_transition_width)
				linear_model_width_weight 			= 1-full_width_weight
				self.width_vec[i] 					= full_width_weight * width_full_width + linear_model_width_weight * width_linear_model
			else:
				raise ValueError('Invalid value of station_x occured')

			# Draft
			if station_x <= self.max_draft_aft and station_x >= 0:
				dimless_pos_transition_aft_draft 	= station_x/self.max_draft_aft
				draft_transom_weight 				= cos_transition_func(dimless_pos_transition_aft_draft)
				draft_max_weight 					= 1 - draft_transom_weight
				self.draft_vec[i] 					= draft_transom_weight * self.draft_transom + draft_max_weight * self.draft_max
			elif station_x < self.max_draft_fwd:
				self.draft_vec[i] 					= self.draft_max
			elif station_x <= self.l_wl:
				dimless_pos_transition_fwd_draft 	= (station_x - self.max_draft_fwd) / (self.l_wl - self.max_draft_fwd)
				draft_fwd_weight_superelliptic		= ((1-dimless_pos_transition_fwd_draft**self.draft_dist_el_exp_fwd))**(1/self.draft_dist_el_exp_fwd)
				self.draft_vec[i]					= self.draft_max * draft_fwd_weight_superelliptic
			else:
				raise ValueError('Invalid value of station_x occured')

			# Cross-sectional superellipse exponent
			if station_x <= self.const_cs_el_exp_aft and station_x >= 0:
				self.cs_el_exp_vec[i] 				= self.cs_el_exp_aft
			elif station_x < self.const_cs_el_exp_fwd:
				dimless_pos_transition_cs_el 		= (station_x - self.const_cs_el_exp_aft) / (self.const_cs_el_exp_fwd - self.const_cs_el_exp_aft)
				cs_el_aft_weight 					= cos_transition_func(dimless_pos_transition_cs_el)
				cs_el_fwd_weight 					= 1-cs_el_aft_weight
				self.cs_el_exp_vec[i] 				= cs_el_aft_weight * self.cs_el_exp_aft + cs_el_fwd_weight * self.cs_el_exp_fwd
			elif station_x <= self.l_wl:
				self.cs_el_exp_vec[i] 				= self.cs_el_exp_fwd
			else:
				raise ValueError('Invalid value of station_x occured')

			# Above-water flare
			if station_x <= self.const_abv_w_flare_fwd and station_x >= 0:
				self.abv_w_flare_w_vec[i] 			= self.abv_w_flare_w_aft
				self.abv_w_flare_h_vec[i] 			= self.abv_w_flare_h_aft
			elif station_x <= self.l_wl:
				dimless_pos_abv_w_flare 			= (station_x - self.const_abv_w_flare_fwd) / (self.l_wl - self.const_abv_w_flare_fwd)
				abv_w_aft_weight 					= (1-dimless_pos_abv_w_flare**(self.abv_w_flare_dist_se_exp))**(1/self.abv_w_flare_dist_se_exp)
				abv_w_fwd_weight 					= 1 - abv_w_aft_weight
				self.abv_w_flare_w_vec[i] 			= abv_w_aft_weight * self.abv_w_flare_w_aft + abv_w_fwd_weight * 0

				abv_w_flare_h_aft_weight 			= 1 - dimless_pos_abv_w_flare**2 # Quadratic distribution
				abv_w_flare_h_fwd_weight 			= 1 - abv_w_flare_h_aft_weight
				self.abv_w_flare_h_vec[i] 			= abv_w_flare_h_aft_weight * self.abv_w_flare_h_aft + abv_w_flare_h_fwd_weight * self.abv_w_flare_h_fwd
			else:
				raise ValueError('Invalid value of station_x occured')

	def generate_sections(self):
		self.cs_array 						= []
		self.cs_A_vec 						= np.zeros(self.n_cs)

		for i in range(self.n_cs):
			if self.vessel_type == 'fast-ferry':
				if self.edgy_cs:
					cs 									= cross_sections.CrossSection_edgy_superellipse(self.n_flat_sides_subm_edgy_cs, self.n_flat_sides_flare_edgy_cs)
					# cs.n_thetas_half should not be overridden. Will be set in the __init__method of CrossSection_edgy_superellipse
				else:
					cs 									= cross_sections.CrossSection_superellipse()
					cs.n_thetas_half					= self.n_cs_verts_subm_half
			elif self.vessel_type == 'wigley':
				cs 									= cross_sections.CrossSection_wigley()
				cs.n_points_half 					= self.n_cs_verts_subm_half
			cs.set_section_data(self.width_vec[i], self.draft_vec[i], self.cs_el_exp_vec[i])
			cs.generate_verts()
			cs.add_above_water_flared_part(self.abv_w_flare_h_vec[i], self.n_cs_verts_abv_w_flare_half, self.abv_w_flare_w_vec[i])
			cs.add_above_water_top_part(self.h_main_deck)
			self.cs_A_vec[i] 					= cs.get_submerged_area()
			self.cs_array.append(cs)

	def generate_verts(self):
		''' Generate array of 3D vertices for the whole hull '''
		self.n_cs_verts_to_deck_half 		= self.n_cs_verts_subm_half + self.n_cs_verts_abv_w_half
		self.n_verts_total_half				= self.n_cs * self.n_cs_verts_to_deck_half
		self.verts_array 					= np.zeros((self.n_verts_total_half, 3))

		for i in range(self.n_cs):
			start_ind 									= i * self.n_cs_verts_to_deck_half
			end_ind 									= start_ind + self.n_cs_verts_to_deck_half
			self.verts_array[start_ind:end_ind, 0]		= self.station_x_vec[i] * np.ones(self.n_cs_verts_to_deck_half)
			self.verts_array[start_ind:end_ind, 1:3]	= self.cs_array[i].verts_vec_half
		self.verts_generated 				= True



	def generate_faces(self):
		''' Generate triangular mesh faces '''
		self.n_verts_per_face 				= 3
		n_faces_per_cs_half 				= 2 * (self.n_cs_verts_to_deck_half - 1)
		self.n_faces_total_half 			= n_faces_per_cs_half * (self.n_cs - 1)
		self.face_verts_array 				= np.zeros((self.n_faces_total_half, self.n_verts_per_face))

		face_verts_counter 					= 0
		for i in range(self.n_cs - 1):
			start_ind_rear_cs 					= i*self.n_cs_verts_to_deck_half
			start_ind_fwd_cs 					= start_ind_rear_cs + self.n_cs_verts_to_deck_half
			for j in range(self.n_cs_verts_to_deck_half - 1):
				self.face_verts_array[face_verts_counter, 0]	= int(start_ind_rear_cs + j) + 1 #+1 because of 1 indexing in .obj file
				self.face_verts_array[face_verts_counter, 1] 	= int(start_ind_fwd_cs + j + 1) + 1 #+1 because of 1 indexing in .obj file
				self.face_verts_array[face_verts_counter, 2] 	= int(start_ind_fwd_cs + j) + 1 #+1 because of 1 indexing in .obj file
				face_verts_counter 								+= 1
				self.face_verts_array[face_verts_counter, 0] 	= int(start_ind_rear_cs + j) + 1 #+1 because of 1 indexing in .obj file
				self.face_verts_array[face_verts_counter, 1] 	= int(start_ind_rear_cs + j + 1) + 1 #+1 because of 1 indexing in .obj file
				self.face_verts_array[face_verts_counter, 2] 	= int(start_ind_fwd_cs + j + 1) + 1 #+1 because of 1 indexing in .obj file
				face_verts_counter 								+= 1

		# Close stern
		if self.close_mesh:
			# Stern
			self.n_verts_total_half 		+= 1
			new_vert 						= np.zeros(3)
			new_vert[2] 					= self.verts_array[0, 2]
			new_vert 						= np.atleast_2d(new_vert)
			self.verts_array 				= np.concatenate((self.verts_array, new_vert), axis = 0)
			n_transom_faces_half 			= self.n_cs_verts_to_deck_half - 1
			self.n_faces_total_half 		+= n_transom_faces_half
			face_verts_addition 			= np.zeros((n_transom_faces_half, 3))
			self.face_verts_array 			= np.concatenate((self.face_verts_array, face_verts_addition), axis = 0)
			for i in range(self.n_cs_verts_to_deck_half-1):
				self.face_verts_array[face_verts_counter, 0] 	= i + 1 #+1 because of 1 indexing in .obj file
				self.face_verts_array[face_verts_counter, 1] 	= (self.n_verts_total_half - 1) + 1 #+1 because of 1 indexing in .obj file
				self.face_verts_array[face_verts_counter, 2] 	= (i + 1) + 1 #+1 because of 1 indexing in .obj file
				face_verts_counter 								+= 1

			# Top
			#Add a row of new verts at the centerline at deck height
			center_vert_stern_ind 			= self.n_verts_total_half - 1
			n_new_verts_deck				= self.n_cs - 1 # No need to generate a new center vert at the stern
			center_verts_deck 				= np.zeros((n_new_verts_deck, 3))
			for i in range(self.n_cs - 1):
				side_vert_ind 					= (i + 1) * self.n_cs_verts_to_deck_half
				center_verts_deck[i, 0] 		= self.verts_array[side_vert_ind, 0]
				center_verts_deck[i, 2] 		= self.verts_array[side_vert_ind, 2]
			self.verts_array 				= np.concatenate((self.verts_array, center_verts_deck), axis = 0)
			self.n_verts_total_half 		+= n_new_verts_deck

			n_new_faces_deck 				= 2 * n_new_verts_deck #No need to correct by one, since we re-use one vert
			new_faces_deck 					= np.zeros((n_new_faces_deck, 3))
			for i in range(n_new_verts_deck):
				rear_inner_vert_ind 						= int(center_vert_stern_ind + i)
				fwd_inner_vert_ind 							= int(rear_inner_vert_ind + 1)
				rear_outer_vert_ind 						= int(i * self.n_cs_verts_to_deck_half)
				fwd_outer_vert_ind 							= int((i + 1) * self.n_cs_verts_to_deck_half)
				start_ind_new_face 							= int(2*i)
				new_faces_deck[start_ind_new_face, 0] 		= fwd_outer_vert_ind + 1#+1 because of 1 indexing in .obj file
				new_faces_deck[start_ind_new_face, 1] 		= rear_inner_vert_ind + 1#+1 because of 1 indexing in .obj file
				new_faces_deck[start_ind_new_face, 2] 		= rear_outer_vert_ind + 1#+1 because of 1 indexing in .obj file

				new_faces_deck[start_ind_new_face+1, 0] 	= fwd_outer_vert_ind + 1#+1 because of 1 indexing in .obj file
				new_faces_deck[start_ind_new_face+1, 1] 	= fwd_inner_vert_ind + 1#+1 because of 1 indexing in .obj file
				new_faces_deck[start_ind_new_face+1, 2] 	= rear_inner_vert_ind + 1#+1 because of 1 indexing in .obj file
			self.face_verts_array 			= np.concatenate((self.face_verts_array, new_faces_deck), axis = 0)
			self.n_faces_total_half 		+= n_new_faces_deck

		# Reverse direction of normals by reversing order of face vertices within each face
		self.face_verts_array = np.flip(self.face_verts_array, axis = 1)

		self.faces_generated 				= True

	def calc_hydrostatic_properties(self):
		self.hull_mesh 							= mesh.Mesh()
		faces_array 							= self.face_verts_array - 1 #Zero indexed faces_array, in accordance with the convention of my_mesh
		self.hull_mesh.import_from_verts_and_faces_arrays(self.verts_array, faces_array)
		self.hull_mesh.mirror(axis_no = 1)
		self.hull_mesh.chop_at_z0(close_top = False)
		self.hull_mesh.calc_volume_properties()
		if self.hull_mesh.volume < 0:
			self.hull_mesh.reverse_normal_direction()
		self.hull_mesh.calc_volume_properties()
		self.hull_mesh.calc_surface_area()

		self.volume_displacement 				= self.hull_mesh.volume
		self.width_max 								= np.amax(self.width_vec)
		self.cb									= self.volume_displacement / (self.l_wl * self.width_max * self.draft_max)
		self.slenderness_ratio 					= self.l_wl / self.volume_displacement**(1/3)

		self.l_to_b 							= self.l_wl / self.width_max
		self.b_to_t 							= self.width_max / self.draft_max
		draft_transom							= self.draft_vec[0]
		self.Tt_to_T 							= draft_transom / self.draft_max

		self.lcb_absolute						= self.hull_mesh.volume_center[0]
		self.lcb_relative						= self.lcb_absolute / self.l_wl
		self.wetted_surface 					= self.hull_mesh.surface_area

	#------------------------------------------------------------------------------------------------
	# Geometry mesh export method
	#------------------------------------------------------------------------------------------------

	def export_obj(self, output_file_path):
		if self.verts_generated != True:
			self.generate_verts()
		if self.faces_generated != True:
			self.generate_faces()
		outfile 							= open(output_file_path, 'w')
		outfile.write('# Hull mesh exported from the hull_generator module of the FASTSHIPS software by the Department of Marine Technology, NTNU\n')
		outfile.write('#\n')
		outfile.write('o object\n')
		for i in range(self.n_verts_total_half):
			outfile.write('v {:.6f} {:.6f} {:.6f}\n'.format(self.verts_array[i, 0], self.verts_array[i, 1], self.verts_array[i, 2]))
		outfile.write('g object_group\n')
		for i in range(self.n_faces_total_half):
			outfile.write('f ')
			for j in range(self.n_verts_per_face):
				outfile.write('{:d} '.format(int(self.face_verts_array[i, j])))
			outfile.write('\n')
		outfile.close()

	#------------------------------------------------------------------------------------------------
	# Figures export methods
	#------------------------------------------------------------------------------------------------

	def export_waterlines(self, output_directory):
		'''Export figure of submerged hull waterlines'''
		z_vals_waterlines_min           = -1 * np.amax(self.draft_vec)
		z_vals_waterlines_max           = 0
		n_waterlines                    = 10


		mirror_plot                     = True
		linesplan_line_width            = 0.25
		line_colour_hull                = 'dimgray'#'dimgray'
		label_font_size                 = 18
		ticks_font_size 				= 20

		# Waterlines plot
		y_interp_params                 = np.zeros((self.n_verts_total_half, 2))
		y_interp_params[:, 0]           = self.verts_array[:, 0]
		y_interp_params[:, 1]           = self.verts_array[:, 2]
		y_interp_values                 = self.verts_array[:, 1]

		y_interp                        = LinearNDInterpolator(y_interp_params, y_interp_values, rescale = True)

		
		z_vals_waterlines               = np.linspace(z_vals_waterlines_min, z_vals_waterlines_max, n_waterlines)

		x_plot_waterlines_min           = 0
		x_plot_waterlines_max           = np.amax(self.verts_array[:, 0])
		n_x_plot_waterlines             = 1000
		x_plot_waterlines               = np.linspace(x_plot_waterlines_min, x_plot_waterlines_max, n_x_plot_waterlines)

		y_vals_waterlines               = np.zeros((n_waterlines, n_x_plot_waterlines))
		for i in range(n_waterlines):
			z_current = z_vals_waterlines[i]
			#print(z_current)
			for j in range(n_x_plot_waterlines):
				x_current 					= x_plot_waterlines[j]
				y_vals_waterlines[i, j]     = y_interp(x_current, z_current)


		fig,ax      					= plt.subplots(figsize = (11.5, 4.1))
		for i in range(n_waterlines):
			ax.plot(x_plot_waterlines/self.l_wl, y_vals_waterlines[i, :]/self.l_wl, color = line_colour_hull, linewidth = linesplan_line_width)
			if mirror_plot:
				ax.plot(x_plot_waterlines/self.l_wl, -y_vals_waterlines[i, :]/self.l_wl, color = line_colour_hull, linewidth = linesplan_line_width)
		ax.axis('equal')
		ax.set_xlabel('x / L [-]', fontsize = label_font_size)
		ax.set_ylabel('y / L [-]', fontsize = label_font_size)
		ax.tick_params(axis='both', which='major', labelsize = ticks_font_size)
		ax.grid(alpha = 0.3)
		fig.tight_layout()
		now                  			= datetime.now()
		now_string           			= now.strftime("%Y%m%d")
		fig_name       					= now_string + '_waterlines'
		fig.savefig(output_directory + '/' + fig_name + '.png', dpi = 900)
		fig.savefig(output_directory + '/' + fig_name + '.pdf')
	
	def export_sections(self, output_directory, plot_section_decimation_factor = 50):
		'''Export illustration of hull sections. Decimate longitudinally by plot_section_decimation_section.'''
		front_sections_line_width       = 0.5
		figure_size_narrow              = (7, 5)
		line_colour_hull                = 'dimgray'
		mirror_hull_plot 				= True
		label_font_size                 = 18
		ticks_font_size 				= 20

		fig,ax      = plt.subplots(figsize = figure_size_narrow)
		for i in range(self.n_cs):
			if i%plot_section_decimation_factor == 0:
				start_ind               	= i * self.n_cs_verts_to_deck_half
				end_ind                   	= start_ind + self.n_cs_verts_to_deck_half
				ax.plot(self.verts_array[start_ind:end_ind, 1] / self.l_wl, self.verts_array[start_ind:end_ind, 2] / self.l_wl, linewidth = front_sections_line_width, color = line_colour_hull)
				if mirror_hull_plot:
					ax.plot(-self.verts_array[start_ind:end_ind, 1] / self.l_wl, self.verts_array[start_ind:end_ind, 2] / self.l_wl, linewidth = front_sections_line_width, color = line_colour_hull)
		ax.axis('equal')
		ax.set_xlabel('x / L [-]', fontsize = label_font_size)
		ax.set_ylabel('z / L [-]', fontsize = label_font_size)
		ax.tick_params(axis='both', which='major', labelsize = ticks_font_size)
		ax.grid(alpha = 0.3)
		fig.tight_layout()
		now                  			= datetime.now()
		now_string           			= now.strftime("%Y%m%d")
		fig_name       					= now_string + '_sections_front'
		fig.savefig(output_directory + '/' + fig_name + '.png', dpi = 900)
		fig.savefig(output_directory + '/' + fig_name + '.pdf')
	
	#------------------------------------------------------------------------------------------------
	# Hull data export method
	#------------------------------------------------------------------------------------------------
	def export_hull_data(self, output_directory, output_file_name = 'hull_data'):
		self.calc_hydrostatic_properties()

		now                  			= datetime.now()
		now_string           			= now.strftime("%Y%m%d")
		outfile_name   					= now_string + '_' + output_file_name
		outpath 						= output_directory + '/' + outfile_name

		outfile 						= open(outpath, 'w')
		outfile.write('Hull data file, exported from the hull_generator module of the FASTSHIPS software by the Department of Marine Technology, NTNU\n')
		outfile.write('\n')
		outfile.write('Main dimensions:\n')
		outfile.write('LWL                                  {:.6f} m\n'.format(self.l_wl))
		outfile.write('B                                    {:.6f} m\n'.format(self.width_max))
		outfile.write('T                                    {:.6f} m\n'.format(self.draft_max))
		outfile.write('\n')
		outfile.write('Hydrostatic properties:\n')
		outfile.write('Volume displacement                  {:.6f} m^3\n'.format(self.volume_displacement))
		outfile.write('Block coefficient                    {:.6f}\n'.format(self.cb))
		outfile.write('Slenderness ratio                    {:.6f}\n'.format(self.slenderness_ratio))
		outfile.write('LCB, absolute                        {:.6f} m\n'.format(self.lcb_absolute))
		outfile.write('LCB, relative                        {:.6f} \n'.format(self.lcb_relative))
		outfile.write('Wetted surface                       {:.6f} m^2\n'.format(self.wetted_surface))
		outfile.write('L/B                                  {:.6f}\n'.format(self.l_to_b))
		outfile.write('B/T                                  {:.6f}\n'.format(self.b_to_t))
		outfile.write('T_transom/T                          {:.6f}\n'.format(self.Tt_to_T))
		outfile.close()

		