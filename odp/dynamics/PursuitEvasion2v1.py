import heterocl as hcl
import numpy as np
import math

class PursuitEvasion2v1:

    def __init__(self, x = [0,0,0,0,0,0], pursuer_R=0.2, evader_R=0.2, bdry_width=2, bdry_length=2,vel_e=0.2,
                  vel_p=0.2, obs_width=0.4, obs_length=0.4, dMax=0, uMode="min", dMode="max"):
        # Environemnt Variables
        self.pursuer_R = pursuer_R
        self.evader_R = evader_R
        self.collison_R = pursuer_R+evader_R
        
        self.vel_p = vel_p
        self.vel_e = vel_e

        self.bdry_width = bdry_width
        self.bdry_length = bdry_length
        
        self.bdry_l = -bdry_width/2
        self.bdry_r = bdry_width/2
        self.bdry_up = bdry_length/2
        self.bdry_dn = -bdry_length/2

        self.obs_width = obs_width
        self.obs_length = obs_length

        self.bdry_l = -obs_width/2
        self.bdry_r = obs_width/2
        self.bdry_up = obs_length/2
        self.bdry_dn = -obs_length/2

        self.dMax = vel_e
        self.uMax = vel_p
        self.uMode = uMode
        self.dMode = dMode

    def dynamics(self, t, state, uOpt, dOpt):
        # maximum velocity
        # vA = hcl.scalar(1.0, "vA")
        # vD = hcl.scalar(1.0, "vD")

        xA11_dot = hcl.scalar(0, "xA11_dot")
        xA12_dot = hcl.scalar(0, "xA12_dot")
        xA21_dot = hcl.scalar(0, "xA21_dot")
        xA22_dot = hcl.scalar(0, "xA22_dot")
        xD1_dot = hcl.scalar(0, "xD1_dot")
        xD2_dot = hcl.scalar(0, "xD2_dot")

        xA11_dot[0] = self.vel_p * uOpt[0]
        xA12_dot[0] = self.vel_p * uOpt[1]
        xA21_dot[0] = self.vel_p * uOpt[2] 
        xA22_dot[0] = self.vel_p * uOpt[3] 
        xD1_dot[0] = self.vel_e * dOpt[0]
        xD2_dot[0] = self.vel_e * dOpt[1]

        return xA11_dot[0], xA12_dot[0], xA21_dot[0], xA22_dot[0], xD1_dot[0], xD2_dot[0]


    def opt_ctrl(self, t, state, spat_deriv):
        """
        :param t: time t
        :param state: tuple of coordinates
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return:
        """
        # In 2v1AttackerDefender, a(t) = [a1, a2, a3, a4]^T
        opt_a1 = hcl.scalar(0, "opt_a1")
        opt_a2 = hcl.scalar(0, "opt_a2")
        opt_a3 = hcl.scalar(0, "opt_a3")
        opt_a4 = hcl.scalar(0, "opt_a4")        
        # Just create and pass back, even though they're not used
        # in3 = hcl.scalar(0, "in3")
        # in4 = hcl.scalar(0, "in4")
        # declare the hcl scalars for relevant spat_derivs
        deriv1 = hcl.scalar(0, "deriv1")
        deriv2 = hcl.scalar(0, "deriv2")
        deriv3 = hcl.scalar(0, "deriv3")
        deriv4= hcl.scalar(0, "deriv4")
        deriv1[0] = spat_deriv[0]
        deriv2[0] = spat_deriv[1]
        deriv3[0] = spat_deriv[2]
        deriv4[0] = spat_deriv[3]
        ctrl_len1 = hcl.sqrt(deriv1[0] * deriv1[0] + deriv2[0] * deriv2[0])     
        ctrl_len2 = hcl.sqrt(deriv3[0] * deriv3[0] + deriv4[0] * deriv4[0])
        if self.uMode == "min":
            with hcl.if_(ctrl_len1 == 0):
                opt_a1[0] = 0.0
                opt_a2[0] = 0.0
            with hcl.else_():
                opt_a1[0] = -deriv1[0] / ctrl_len1
                opt_a2[0] = -deriv2[0] / ctrl_len1
            with hcl.if_(ctrl_len2 == 0):
                opt_a3[0] = 0.0
                opt_a4[0] = 0.0
            with hcl.else_():
                opt_a3[0] = -deriv3[0] / ctrl_len2
                opt_a4[0] = -deriv4[0] / ctrl_len2
        else:
            with hcl.if_(ctrl_len1 == 0):
                opt_a1[0] = 0.0
                opt_a2[0] = 0.0
            with hcl.else_():
                opt_a1[0] = deriv1[0] / ctrl_len1
                opt_a2[0] = deriv2[0] / ctrl_len1
            with hcl.if_(ctrl_len2 == 0):
                opt_a3[0] = 0.0
                opt_a4[0] = 0.0
            with hcl.else_():
                opt_a3[0] = deriv3[0] / ctrl_len2
                opt_a4[0] = deriv4[0] / ctrl_len2
        # return 3, 4 even if you don't use them
        return opt_a1[0], opt_a2[0], opt_a3[0], opt_a4[0]

    def opt_dstb(self, t, state, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        # Graph takes in 4 possible inputs, by default, for now
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        # Just create and pass back, even though they're not used
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        # the same procedure in opt_ctrl
        deriv1 = hcl.scalar(0, "deriv1")
        deriv2 = hcl.scalar(0, "deriv2")
        deriv1[0] = spat_deriv[4]
        deriv2[0] = spat_deriv[5]
        dstb_len = hcl.sqrt(deriv1[0] * deriv1[0] + deriv2[0] * deriv2[0])
        # with hcl.if_(self.dMode == "max"):
        if self.dMode == 'max':
            with hcl.if_(dstb_len == 0):
                d1[0] = 0.0
                d2[0] = 0.0
            with hcl.else_():
                d1[0] = deriv1[0] / dstb_len
                d2[0] = deriv2[0] / dstb_len
        else:
            with hcl.if_(dstb_len == 0):
                d1[0] = 0.0
                d2[0] = 0.0
            with hcl.else_():
                d1[0] = -deriv1[0]/ dstb_len
                d2[0] = -deriv2[0] / dstb_len

        return d1[0], d2[0], d3[0], d4[0]

        # The below function can have whatever form or parameters users want
        # These functions are not used in HeteroCL program, hence is pure Python code and
        # can be used after the value function has been obtained.

    def optCtrl_inPython(self, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal control of two attackers
        """
        opt_a1 = self.uMax
        opt_a2 = self.uMax
        opt_a3 = self.uMax
        opt_a4 = self.uMax
        deriv1 = spat_deriv[0]
        deriv2 = spat_deriv[1]
        deriv3 = spat_deriv[2]
        deriv4 = spat_deriv[3]
        ctrl_len1 = np.sqrt(deriv1*deriv1 + deriv2*deriv2)
        ctrl_len2 = np.sqrt(deriv3*deriv3 + deriv4*deriv4)
        # The initialized control only change sign in the following cases
        if self.uMode == "min":
            if ctrl_len1 == 0:
                opt_a1 = 0.0
                opt_a2 = 0.0
            else:
                opt_a1 = -deriv1 / ctrl_len1
                opt_a2 = -deriv2 / ctrl_len1
            if ctrl_len2 == 0:
                opt_a3 = 0.0 
                opt_a4 = 0.0
            else:
                opt_a3 = -deriv3 / ctrl_len2
                opt_a4 = -deriv4 / ctrl_len2
        else:
            if ctrl_len1 == 0:
                opt_a1 = 0.0
                opt_a2 = 0.0
            else:
                opt_a1 = deriv1 / ctrl_len1
                opt_a2 = deriv2 / ctrl_len1
            if ctrl_len2 == 0:
                opt_a3 = 0.0 
                opt_a4 = 0.0
            else:
                opt_a3 = deriv3 / ctrl_len2
                opt_a4 = deriv4 / ctrl_len2
        return (opt_a1, opt_a2, opt_a3, opt_a4)
    
    def optDstb_inPython(self, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal control of the defender (disturbances)
        """
        opt_d1 = self.dMax
        opt_d2 = self.dMax
        deriv5 = spat_deriv[4]
        deriv6 = spat_deriv[5]
        dstb_len = np.sqrt(deriv5*deriv5 + deriv6*deriv6)
        # The initialized control only change sign in the following cases
        if self.dMode == "max":
            if dstb_len == 0:
                opt_d1 = 0.0
                opt_d2 = 0.0
            else:
                opt_d1 = self.vel_e*deriv5 / dstb_len
                opt_d2 = self.vel_e*deriv6 / dstb_len
        else:
            if dstb_len == 0:
                opt_d1 = 0.0
                opt_d2 = 0.0
            else:
                opt_d1 = -self.vel_e*deriv5 / dstb_len
                opt_d2 = -self.vel_e*deriv6 / dstb_len
        return (opt_d1, opt_d2)

    def capture_set1(self, grid, capture_radius, mode):
        data = np.power(grid.vs[0] - grid.vs[4], 2) + np.power(grid.vs[1] -grid.vs[5], 2)
        if mode == "capture":
            return np.sqrt(data) - capture_radius
        if mode == "escape":
            return capture_radius - np.sqrt(data)

    def capture_set2(self, grid, capture_radius, mode):
        data = np.power(grid.vs[2] - grid.vs[4], 2) + np.power(grid.vs[3] -grid.vs[5], 2)
        if mode == "capture":
            return np.sqrt(data) - capture_radius
        if mode == "escape":
            return capture_radius - np.sqrt(data)