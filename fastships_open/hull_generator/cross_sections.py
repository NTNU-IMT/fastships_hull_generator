# © 2023, NTNU
# Author: John Martin Kleven Godø <john.martin.godo@ntnu.no>
# This code is licenced under the GNU General Public License v3.0

import numpy as np
from scipy.special import gamma as gamma_func


class CrossSection_wigley():
	def __init__(self):
		a = 0
		self.n_points_half 			= 501

	def set_section_data(self, width, draft, foo):
		self.halfwidth 				= 0.5 * width
		self.draft 					= draft

	def generate_verts(self):
		self.verts_vec_half			= np.zeros((self.n_points_half, 2))
		for i in range(self.n_points_half):
			dimless_dist_from_top		= i / (self.n_points_half-1)
			self.verts_vec_half[i, 0] 	= self.halfwidth * (1 - dimless_dist_from_top**2)
			self.verts_vec_half[i, 1] 	= -1 * dimless_dist_from_top * self.draft

	def add_above_water_flared_part(self, height, n_verts_abv_w, delta_width_flare):
		self.abv_w_height 					= height
		self.n_verts_abv_w 					= n_verts_abv_w
		self.abv_w_flare_verts_vec_half 			= np.zeros((self.n_verts_abv_w, 2))
		self.abv_w_flare_verts_vec_half[:, 1] 	= np.linspace(self.abv_w_height, 0, self.n_verts_abv_w)
		halfwidth_wl 						= self.verts_vec_half[0, 0]
		for i in range(self.n_verts_abv_w):
			self.abv_w_flare_verts_vec_half[i, 0] 	= halfwidth_wl + delta_width_flare*(self.abv_w_flare_verts_vec_half[i, 1]/self.abv_w_height)**2
		self.verts_vec_half 				= np.concatenate((self.abv_w_flare_verts_vec_half, self.verts_vec_half), axis = 0)

	def add_above_water_top_part(self, height_to_main_deck):
		self.abv_w_top_part_verts 			= np.zeros((2, 2))
		self.abv_w_top_part_verts[-1, :] 	= self.abv_w_flare_verts_vec_half[0, :]
		self.abv_w_top_part_verts[0, 0] 	= self.abv_w_flare_verts_vec_half[0, 0]
		self.abv_w_top_part_verts[0, 1] 	= height_to_main_deck
		self.verts_vec_half 				= np.concatenate((self.abv_w_top_part_verts, self.verts_vec_half), axis = 0)

	def get_submerged_area(self):
		area_one_side_submerged 			= 2/3 * self.halfwidth * self.draft
		area_both_sides_submerged 			= 2 * area_one_side_submerged
		self.area 							= area_both_sides_submerged
		return self.area

class CrossSection_superellipse():
	def __init__(self):
		a = 0
		self.n_thetas_half			= 501

	def set_section_data(self, width, draft, se_exp):
		self.a 						= 0.5*width
		self.b  					= draft
		self.se_exp 				= se_exp #Superellipse exponent

	def generate_verts(self):
		self.verts_vec_half 		= np.zeros((self.n_thetas_half, 2))
		self.r_vec_half 			= np.zeros(self.n_thetas_half)

		self.theta_vec_half         = np.linspace(0, -0.5*np.pi, self.n_thetas_half)
		if self.a != 0 and self.b != 0:
			for i in range(self.n_thetas_half):
				self.r_vec_half[i] 			= self.a*self.b/(np.absolute(self.b*np.cos(self.theta_vec_half[i]))**self.se_exp + np.absolute(self.a*np.sin(self.theta_vec_half[i]))**self.se_exp)**(1/self.se_exp)
				self.verts_vec_half[i, 0]   = self.r_vec_half[i]*np.cos(self.theta_vec_half[i])
				self.verts_vec_half[i, 1]   = self.r_vec_half[i]*np.sin(self.theta_vec_half[i])

	def get_submerged_area(self):
		a 									= self.a
		b 									= self.b
		area_whole_superellipse 			= 4*a*b*(gamma_func(1+1/self.se_exp))**2/(gamma_func(1+2/self.se_exp))
		self.area							= 0.5*area_whole_superellipse
		return self.area


	def generate_area_vals_widthLocked(self):
		self.n_area_vals_widthLocked 			= 10000
		self.area_vals_widthLocked 				= np.zeros(self.n_area_vals_widthLocked)

		draft_min_widthLocked 					= 0
		draft_max_widthLocked 					= 100*(2*self.a)
		self.b_vals_widthLocked                 = np.linspace(0, draft_max_widthLocked, self.n_area_vals_widthLocked)

		for i in range(self.n_area_vals_widthLocked):
			a 									= self.a
			b 									= self.b_vals_widthLocked[i]
			area_whole_superellipse 			= 4*a*b*(gamma_func(1+1/self.se_exp))**2/(gamma_func(1+2/self.se_exp))
			self.area_vals_widthLocked[i] 		= 0.5*area_whole_superellipse

	def interp_area_draft(self, area_target):
		draft 									= np.interp(area_target, self.area_vals_widthLocked, self.b_vals_widthLocked)
		return draft

	def set_section_data_width_area(self, width, area_target, se_exp):
		self.a									= 0.5*width
		self.se_exp 							= se_exp

		self.generate_area_vals_widthLocked()
		draft 									= self.interp_area_draft(area_target)
		self.b 									= draft

	def mod_to_min_thickness(self, min_thickness):
		min_offset = 0.5*min_thickness
		for i in range(self.n_thetas_half):
			if self.verts_vec_half[i, 0] < min_offset:
				self.verts_vec_half[i, 0] = 0
				self.verts_vec_half[i, 1] = self.verts_vec_half[i-1, 1]

	def add_above_water_flared_part(self, height, n_verts_abv_w, delta_width_flare):
		self.abv_w_height 					= height
		self.n_verts_abv_w 					= n_verts_abv_w
		self.abv_w_flare_verts_vec_half 			= np.zeros((self.n_verts_abv_w, 2))
		self.abv_w_flare_verts_vec_half[:, 1] 	= np.linspace(self.abv_w_height, 0, self.n_verts_abv_w)
		halfwidth_wl 						= self.verts_vec_half[0, 0]
		for i in range(self.n_verts_abv_w):
			self.abv_w_flare_verts_vec_half[i, 0] 	= halfwidth_wl + delta_width_flare*(self.abv_w_flare_verts_vec_half[i, 1]/self.abv_w_height)**2
		self.verts_vec_half 				= np.concatenate((self.abv_w_flare_verts_vec_half, self.verts_vec_half), axis = 0)

	def add_above_water_top_part(self, height_to_main_deck):
		self.abv_w_top_part_verts 			= np.zeros((2, 2))
		self.abv_w_top_part_verts[-1, :] 	= self.abv_w_flare_verts_vec_half[0, :]
		self.abv_w_top_part_verts[0, 0] 	= self.abv_w_flare_verts_vec_half[0, 0]
		self.abv_w_top_part_verts[0, 1] 	= height_to_main_deck
		self.verts_vec_half 				= np.concatenate((self.abv_w_top_part_verts, self.verts_vec_half), axis = 0)


class CrossSection_edgy_superellipse(CrossSection_superellipse):
	def __init__(self, n_flat_sides_subm_edgy_cs, n_flat_sides_flare):
		super().__init__()
		self.n_flat_sides_subm_half			= n_flat_sides_subm_edgy_cs
		self.n_flat_sides_flare				= n_flat_sides_flare
		self.n_thetas_half 					= self.n_flat_sides_subm_half + 1

	def get_submerged_area(self):
		zero_vert 							= np.zeros(2)
		zero_vert 							= np.atleast_2d(zero_vert)
		self.area 							= 0
		for i in range(self.n_thetas_half - 1):
			vert_1 							= self.verts_vec_half[i] - zero_vert
			vert_2 							= self.verts_vec_half[i + 1] - zero_vert
			self.area 						+= 0.5*np.absolute(np.cross(vert_2, vert_1))
		return self.area

	def add_above_water_flared_part(self, height, n_verts_abv_w, delta_width_flare):
		self.abv_w_height 					= height
		print('Discarding n_verts_abv_w and overriding to a value based on self.n_flat_sides_flare')
		self.n_verts_abv_w 					= self.n_flat_sides_flare + 1
		self.abv_w_flare_verts_vec_half 			= np.zeros((self.n_verts_abv_w, 2))
		self.abv_w_flare_verts_vec_half[:, 1] 	= np.linspace(self.abv_w_height, 0, self.n_verts_abv_w)
		halfwidth_wl 						= self.verts_vec_half[0, 0]
		for i in range(self.n_verts_abv_w):
			self.abv_w_flare_verts_vec_half[i, 0] 	= halfwidth_wl + delta_width_flare*(self.abv_w_flare_verts_vec_half[i, 1]/self.abv_w_height)**2
		self.verts_vec_half 				= np.concatenate((self.abv_w_flare_verts_vec_half, self.verts_vec_half), axis = 0)

	def generate_area_vals_widthLocked(self):
		raise ValueError('Needs implementation. Current version (from superclass) is hardcoded to calculate the area by itself. If changed into using the method get_submerged_area then this could probably be used directly')
