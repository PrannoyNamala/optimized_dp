import heterocl as hcl
import numpy as np
import math

class PursuitEvasion1v1:

    def __init__(self, x = [0,0,0,0], pursuer_R=0.2, evader_R=0.2, bdry_width=2, bdry_length=2,vel_e=0.2,
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

        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode

    def dynamics(self, t, state, uOpt, dOpt):
        # maximum velocity
        # vA = hcl.scalar(1.0, "vA")
        # vD = hcl.scalar(1.0, "vD")

        xA1_dot = hcl.scalar(0, "xA1_dot")
        xA2_dot = hcl.scalar(0, "xA2_dot")
        xD1_dot = hcl.scalar(0, "xD1_dot")
        xD2_dot = hcl.scalar(0, "xD2_dot")

        xA1_dot[0] = self.vel_p * uOpt[0]
        xA2_dot[0] = self.vel_p * uOpt[1]
        xD1_dot[0] = self.vel_e * dOpt[0]
        xD2_dot[0] = self.vel_e * dOpt[1]

        return xA1_dot[0], xA2_dot[0], xD1_dot[0], xD2_dot[0]

    def opt_ctrl(self, t, state, spat_deriv):
        """
        :param t: time t
        :param state: tuple of coordinates
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return:
        """
        # In 1v1AttackerDefender, a(t) = [a1, a2]^T
        opt_a1 = hcl.scalar(0, "opt_a1")
        opt_a2 = hcl.scalar(0, "opt_a2")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        # declare the hcl scalars for relevant spat_derivs
        deriv1 = hcl.scalar(0, "deriv1")
        deriv2 = hcl.scalar(0, "deriv2")
        deriv1[0] = spat_deriv[0]
        deriv2[0] = spat_deriv[1]
        ctrl_len = hcl.sqrt(deriv1[0] * deriv1[0] + deriv2[0] * deriv2[0])        
        if self.uMode == "min":
            with hcl.if_(ctrl_len == 0):
                opt_a1[0] = 0.0
                opt_a2[0] = 0.0
            with hcl.else_():
                opt_a1[0] = -1.0 * deriv1[0] / ctrl_len
                opt_a2[0] = -1.0 * deriv2[0] / ctrl_len
        else:
            with hcl.if_(ctrl_len == 0):
                opt_a1[0] = 0.0
                opt_a2[0] = 0.0
            with hcl.else_():
                opt_a1[0] = deriv1[0] / ctrl_len
                opt_a2[0] = deriv2[0] / ctrl_len
        # return 3, 4 even if you don't use them
        return opt_a1[0], opt_a2[0], in3[0], in4[0]

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
        deriv1[0] = spat_deriv[2]
        deriv2[0] = spat_deriv[3]
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
                d1[0] = -1 * deriv1[0]/ dstb_len
                d2[0] = -1 * deriv2[0] / dstb_len

        return d1[0], d2[0], d3[0], d4[0]
    
    def capture_set(self, grid, mode):
        # using meshgrid
        xa, ya, xd, yd = np.meshgrid(grid.grid_points[0], grid.grid_points[1],
                                     grid.grid_points[2], grid.grid_points[3], indexing='ij')
        data = np.power(xa - xd, 2) + np.power(ya - yd, 2)
        if mode == "capture":
            return np.sqrt(data) - self.collison_R 
        if mode == "escape":
            return self.collison_R  - np.sqrt(data)

    def optCtrl_inPython(self, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal control of the attacker
        """
        opt_a1 = self.vel_p
        opt_a2 = self.vel_p
        deriv1 = spat_deriv[0]
        deriv2 = spat_deriv[1]
        ctrl_len = np.sqrt(deriv1*deriv1 + deriv2*deriv2)
        # The initialized control only change sign in the following cases
        if self.uMode == "min":
            if ctrl_len == 0:
                opt_a1 = 0.0
                opt_a2 = 0.0
            else:
                opt_a1 = -deriv1 / ctrl_len
                opt_a2 = -deriv2 / ctrl_len
        else:
            if ctrl_len == 0:
                opt_a1 = 0.0
                opt_a2 = 0.0
            else:
                opt_a1 = deriv1 / ctrl_len
                opt_a2 = deriv2 / ctrl_len
        return (opt_a1, opt_a2)
    
    def optDstb_inPython(self, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal control of the defender (disturbances)
        """
        opt_d1 = self.dMax
        opt_d2 = self.dMax
        deriv3 = spat_deriv[2]
        deriv4 = spat_deriv[3]
        dstb_len = np.sqrt(deriv3*deriv3 + deriv4*deriv4)
        # The initialized control only change sign in the following cases
        if self.dMode == "max":
            if dstb_len == 0:
                opt_d1 = 0.0
                opt_d2 = 0.0
            else:
                opt_d1 = self.vel_e*deriv3 / dstb_len
                opt_d2 = self.vel_e*deriv4 / dstb_len
        else:
            if dstb_len == 0:
                opt_d1 = 0.0
                opt_d2 = 0.0
            else:
                opt_d1 = -self.vel_e*deriv3 / dstb_len
                opt_d2 = -self.vel_e*deriv4 / dstb_len
        return (opt_d1, opt_d2)

