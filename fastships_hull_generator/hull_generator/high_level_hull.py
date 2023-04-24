# © 2023, NTNU
# Author: John Martin Kleven Godø <john.martin.godo@ntnu.no>
# This code is licenced under the GNU General Public License v3.0

import numpy as np
import hull_generator.hull as hull
import fastships_hull_generator.mesh_tools.mesh as mesh





class HighLevelHull1(hull.Hull):
	''' Create a hull based on high-level input in the form of the following parameters:
	- LWL
	- L/B ratio
	- B/T ratio
	- Ratio of depth at transom to max depth.

	Standard settings are defined for all other parameters necessary to define the hull. These
	are equal to those used when generating the NTNU FF1 hull. '''
	def __init__(self, l_wl, l_to_b, b_to_t, Tt_to_T):
		self.l_wl 								= l_wl
		self.l_to_b 							= l_to_b
		self.b_to_t 							= b_to_t
		self.Tt_to_T 							= Tt_to_T
		super().__init__()
		self.set_standard_settings()
		self.generate_hull()

	def set_standard_settings(self):
		self.edgy_cs 							= False
		# Resolution
		n_cs 									= 1001
		n_cs_verts_subm_full 					= 101
		self.set_resolution(n_cs, n_cs_verts_subm_full)
		# Length
		self.set_main_dimensions(self.l_wl)
		# Width settings
		width_stern 							= self.l_wl / self.l_to_b
		entrance_angle 							= 4 * np.pi / 180
		const_width_fwd_rel 					= 0.375
		self.set_width_data(width_stern, entrance_angle, const_width_fwd_rel)
		# Draft settings
		draft_max 								= width_stern / self.b_to_t
		draft_transom 							= draft_max * self.Tt_to_T
		max_draft_aft_rel 						= 0.5
		max_draft_fwd_rel 						= 0.85
		draft_dist_el_exp_fwd 					= 3
		self.set_draft_data(draft_max, draft_transom, max_draft_aft_rel, max_draft_fwd_rel, draft_dist_el_exp_fwd)
		# Cross-sectional shape data
		cs_el_exp_aft							= 4
		cs_el_exp_fwd 							= 1.5
		const_cs_el_exp_aft_rel 				= 0.25
		const_cs_el_exp_fwd_rel					= 0.9
		self.set_cs_el_exp_data(cs_el_exp_aft, cs_el_exp_fwd, const_cs_el_exp_aft_rel, const_cs_el_exp_fwd_rel)
		# Above-water data
		n_cs_verts_abv_w_full            		= 100
		abv_w_flare_h_aft                		= 1.5 * self.l_wl / 40
		abv_w_flare_h_fwd                		= 2.0 * self.l_wl / 40
		abv_w_flare_w_aft                		= 0.3 * self.l_wl / 40
		const_abv_w_flare_fwd_rel        		= 0.5
		abv_w_flare_dist_se_exp          		= 2
		h_main_deck                      		= 2.5 * self.l_wl / 40
		self.set_abv_w_data(n_cs_verts_abv_w_full, abv_w_flare_h_aft, abv_w_flare_h_fwd, abv_w_flare_w_aft, const_abv_w_flare_fwd_rel, abv_w_flare_dist_se_exp, h_main_deck)


	def generate_hull(self):
		self.generate_long_property_distributions()
		self.generate_sections()
		self.generate_verts()
		self.generate_faces()

class HighLevelHull2(HighLevelHull1):
	''' Create a hull based on high-level input in the form of the following parameters:
	- Volume displacement
	- L/B ratio
	- B/T ratio
	- Ratio of depth at transom to max depth.

	Standard settings are defined for all other parameters necessary to define the hull. These
	are equal to those used when generating the NTNU FF1 hull. '''
	def __init__(self, volume_displacement, l_to_b, b_to_t, Tt_to_T):
		self.volume_displacement				= volume_displacement
		l_wl_start 								= 1.0
		super().__init__(l_wl_start, l_to_b, b_to_t, Tt_to_T)

	def scale_to_displacement(self):
		self.hull_mesh 							= mesh.Mesh()
		faces_array 							= self.face_verts_array - 1 #Zero indexed faces_array, in accordance with the convention of my_mesh
		self.hull_mesh.import_from_verts_and_faces_arrays(self.verts_array, faces_array)
		self.hull_mesh.mirror(axis_no = 1)
		self.hull_mesh.chop_at_z0()
		self.hull_mesh.calc_volume_properties()
		if self.hull_mesh.volume < 0:
			self.hull_mesh.reverse_normal_direction()
		self.hull_mesh.calc_volume_properties()
		scale_factor 							= (self.volume_displacement / self.hull_mesh.volume)**(1/3)
		scale_vector 							= scale_factor * np.ones(3)
		scale_origin 							= np.zeros(3)
		self.hull_mesh.scale(scale_vector, scale_origin)
		self.hull_mesh.calc_volume_properties()
		self.verts_array 						*= scale_factor
		self.l_wl								*= scale_factor
		self.draft_vec 							*= scale_factor
		self.draft_max 							*= scale_factor
		self.width_vec 							*= scale_factor
		
