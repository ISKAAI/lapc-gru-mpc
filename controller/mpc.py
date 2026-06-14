import cvxpy as cp
import numpy as np


class MPCController:


    def __init__(self, model, mpc_cfg):
        self.model = model       
        self.cfg = mpc_cfg       

        self.n = model.Ad.shape[0]  
        self.m = 1                  
        self.Np = mpc_cfg.Np
        self.Nc = mpc_cfg.Nc
        self.Nd = mpc_cfg.Nd     

        self.Q = mpc_cfg.Q
        self.R = mpc_cfg.R
        self.P = mpc_cfg.P
        self.S = mpc_cfg.S
        self.delta_max = mpc_cfg.delta_max
        self.ddelta_max = mpc_cfg.ddelta_max

    # ───────────────────────── 核心 ─────────────────────────
    def solve(self, x0, kappa_seq, vx, delta_prev=0.0, pending=None):
        n,m,Np,Nc= self.n, self.m, self.Np, self.Nc
        self.model.update_vx(vx)

        Ad = self.model.Ad
        Bd = self.model.Bd
        x_pred = x0.copy()
        if pending is not None:
            for i in range(self.Nd):
                x_pred = Ad @ x_pred + Bd[:,0]*pending[i] +Bd[:,1] * (vx * kappa_seq[i])

        x = cp.Variable((n,Np+1), name='x')
        u = cp.Variable((1,Nc), name='u') 

        constraints = [ x[:,0] == x_pred ]
        cost = 0
        for k in range (Np):

            psi_des = vx * kappa_seq[k]
            uk = u[0,min(k,self.Nc-1)]
            constraints += [ x[:,k+1] == Ad @ x[:,k] + Bd[:,0] * uk + Bd[:,1] * psi_des]
            cost += cp.quad_form(x[:,k],self.Q) + self.R[0,0] * cp.square(uk)
            
            constraints += [cp.abs(uk) <= self.delta_max]

            if k < self.Nc:                     
                    du = uk - delta_prev if k == 0 else u[0,k] - u[0,k-1]
                    constraints += [ cp.abs(du) <= self.ddelta_max ]
                    cost += self.S * cp.square(du)

        cost += cp.quad_form(x[:,Np],self.P)
        objective = cp.Minimize(cost)
        prob = cp.Problem(objective=objective,constraints=constraints)
        prob.solve(solver = cp.OSQP,max_iter = 20000, polish = True)
        if prob.status not in ('optimal','optimal_inaccurate'):
             return delta_prev
        return u.value[0,0]


