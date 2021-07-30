import numpy as np
from cvxopt import matrix, solvers, sparse
from scipy.sparse import dia_matrix, diags
from scipy import interpolate
class isotonic_regressor:
    def __init__(self, n_gridpoints, isotonic=1, convexity=0):
        self.isotonic = isotonic
        self.convexity = convexity
        self.n_gridpoints = n_gridpoints
        self.fx = lambda x:np.zeros(len(x))
        self.dfx = lambda x:np.zeros(len(x))
        self.dfx2 = lambda x:np.zeros(len(x))


    def spline_penalty(self, grid_points):
        #calculate a spline penalty
        #formula can be found at http://data.princeton.edu/eco572/smoothing.pdf
        #Smoothing and Non-Parametric Regression
        #German Rodriguez grodri@princeton.edu
        #Spring, 2001

        n = grid_points.shape[0]
        delta_b = np.diff(grid_points)
        W = np.diag(np.ones(n-3) * delta_b[:-2]/6,k=-1) + np.diag(np.ones(n-3) * delta_b[2:]/6,k=1) + np.diag(np.ones(n-2) * (delta_b[:-1]+delta_b[1:]) / 3,k=0)
        delta = np.array(diags([1./delta_b[:-1],-1./delta_b[:-1] - 1./delta_b[1:],1./delta_b[1:]], offsets=[0,1,2], shape=(n-2,n)).todense())
        K = delta.T.dot(np.linalg.lstsq(W, delta, rcond=None)[0])
        return K

    def m_to_M(self, m, boundaries):
        #convert a number m to a vector with linear interpolation between members of the boundaries vector
        diffs = (m - boundaries)
        first = np.where(diffs >= 0)[0][-1]
        second = np.where(diffs <= 0)[0][0]
        delta = boundaries[second] - boundaries[first]
        delta += delta == 0
        M = np.zeros(boundaries.shape)
        M[second] = diffs[first] / delta
        M[first] = 1 - diffs[first] / delta
        return M

    def make_matrices(self, xs, ys, alphas, min_x=None, max_x=None):
        if min_x is None:
            min_x = np.nanmin(xs)
        if max_x is None:
            max_x = np.nanmax(xs)

        m = np.linspace(min_x, max_x, self.n_gridpoints)
        M = np.array([self.m_to_M(x, m) for x in xs])

        # With inequality Δx < 0, rewritten as
        # G*x <= h

        G = []
        if self.isotonic != 0:
            isotonic = float(self.isotonic)
            offsets = np.array([0, 1])
            derivative = -np.array([[-isotonic, isotonic]]).repeat(self.n_gridpoints, axis=0)
            G.append(dia_matrix((derivative.T, offsets), shape=(self.n_gridpoints - 1, self.n_gridpoints)).todense())

        if self.convexity != 0:
            convexity = float(self.convexity)
            offsets = np.array([0, 1, 2])
            derivative = np.array([[-convexity, 2*convexity, -convexity]]).repeat(self.n_gridpoints, axis=0)
            G.append(dia_matrix((derivative.T, offsets), shape=(self.n_gridpoints - 2, self.n_gridpoints)).todense())

        if len(G) == 0:
            G = np.zeros((0,self.n_gridpoints))
        else:
            G = np.vstack(G)

        h = np.zeros(G.shape[0])

        #change to cvxopt format
        G = matrix(G)
        h = matrix(h)

        P = (M.T).dot(M)

        #calculate smoothing penalty
        K = self.spline_penalty(m)

        #add smoothing penalty to objective
        Ps = [matrix(P+alpha*K) for alpha in list(alphas)]

        mu, s = np.mean(ys), np.std(ys)

        q = matrix(M.T.dot(-(ys-mu)/s))
        return Ps, q, G, h, m, M, K

    def get_curve(self, x, y, alpha=0, min_x=None, max_x=None, init_vals=None, P=None, q=None, G=None, h=None, x_grid=None):
        mu, s = np.mean(y), np.std(y)
        if P is None:
            P, q, G, h, x_grid, M, K = self.make_matrices(x, y, [alpha], min_x, max_x)
            P = P[0]
        if init_vals is None:
            ret = solvers.qp(P, q, G = G, h = h, init_vals=x_grid)
        else:
            ret = solvers.qp(P, q, G = G, h = h, init_vals=init_vals)

        fit_y = np.array(ret['x']).flatten()*s + mu
        x_grid = x_grid.flatten()

        x_expand = np.hstack((x_grid, x_grid.max() + 1, x_grid.max() + 2, x_grid.max() + 3, 1e14))
        slope = np.gradient(fit_y, x_grid)[-1]
        y_expand = np.hstack((fit_y, fit_y[-1] + slope, fit_y[-1] + 2 * slope, fit_y[-1] + 3 * slope, fit_y[-1] + 1e14 * slope))

        fx = interpolate.interp1d(x_expand, y_expand)
        dfdx = interpolate.interp1d(x_expand, np.gradient(y_expand, x_expand))
        df2dx2 = interpolate.interp1d(x_expand, np.gradient(np.gradient(y_expand, x_expand), x_expand))
        return x_grid, fit_y, fx, dfdx, df2dx2

    def fit_alphas(self, xs, ys, alphas, min_x = None, max_x=None):
        if isinstance(alphas, float):
            alphas = [alphas]
            
        BICs = []
        Alinear = np.array([np.ones(len(xs)), xs])
        mu, s = np.mean(ys), np.std(ys)

        logH = np.linalg.slogdet(Alinear.dot(Alinear.T))[1] * (1./self.n_gridpoints)

        fit_fx = None
        Ps, q, G, h, x_grid, M, K = self.make_matrices(xs, ys, alphas, min_x, max_x)
        dx = x_grid[1] - x_grid[0]
        maxBIC = -np.inf
        for alpha, P in zip(alphas[::-1], Ps[::-1]):
            print(f'Fitting standard with smoothing penalty:{alpha}                   ',end="\r")
            try:
                x_grid, fit_y, fx, dfdx, df2dx2 = self.get_curve(xs, ys, min_x = min_x, max_x=max_x, init_vals=fit_fx, P=P, q=q, G=G, h=h, x_grid=x_grid)
                LL = np.array(P).dot((fit_y-mu)/s).dot((fit_y-mu)/s) - 2 * np.array(q).flatten().dot((fit_y-mu)/s) #+ ys.dot(ys)
                S = np.linalg.pinv(np.array(P)) + np.diag(np.ones(self.n_gridpoints) * np.exp(-logH))
                BICs = [ 2 * LL - np.linalg.slogdet(S)[1] ] + BICs

                if BICs[0] > maxBIC:
                    maxBIC = BICs[0]
                    best_x_grid, best_fit_y, best_fx, bestdfdx, best_df2dx = x_grid, fit_y, fx, dfdx, df2dx2

            except:
                print(f'fit failed at alpha={alpha}', end='\r')
                BICs = [ -np.inf] + BICs

        self.x_grid = best_x_grid
        self.y_knots = best_fit_y
        self.fx = best_fx
        self.dfx = bestdfdx
        self.dfx2 = best_df2dx
        return BICs

    def predict(self, xs):
        return self.fx(xs)

    def predict_gradient(self,xs):
        return self.dfx(xs)

    def predict_hessian(self,xs):
        return self.dfx2(xs)
