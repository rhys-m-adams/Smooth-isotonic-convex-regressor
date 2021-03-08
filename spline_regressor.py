import numpy as np
from cvxopt import matrix, solvers, sparse
from scipy.sparse import dia_matrix, diags
from scipy import interpolate

def spline_penalty(grid_points, alpha):
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
    A = np.linalg.inv(np.eye(K.shape[0]) - alpha * K)
    return K, 1 - np.mean(A.diagonal())

def m_to_M(m, boundaries):
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

def make_matrices(xs, ys, n_gridpoints, alphas, min_x=None, max_x=None, isotonic=1, convexity=0):
    if min_x is None:
        min_x = np.nanmin(xs)
    if max_x is None:
        max_x = np.nanmax(xs)
    
    m = np.linspace(min_x, max_x, n_gridpoints)
    M = np.array([m_to_M(x, m) for x in xs])

    # With inequality Δx < 0, rewritten as
    # G*x <= h
    
    G = []
    if isotonic != 0:
        isotonic = float(isotonic)
        offsets = np.array([0, 1])
        derivative = -np.array([[-isotonic, isotonic]]).repeat(n_gridpoints, axis=0)
        G.append(dia_matrix((derivative.T, offsets), shape=(n_gridpoints - 1, n_gridpoints)).todense())
    
    if convexity != 0:
        convexity = float(convexity)
        offsets = np.array([0, 1, 2])
        derivative = np.array([[-convexity, 2*convexity, -convexity]]).repeat(n_gridpoints, axis=0)
        G.append(dia_matrix((derivative.T, offsets), shape=(n_gridpoints - 2, n_gridpoints)).todense())
    
    if len(G) == 0:
        G = np.zeros((0,n_gridpoints))
    else:
        G = np.vstack(G)
    
    h = np.zeros(G.shape[0])

    #change to cvxopt format
    G = matrix(G)
    h = matrix(h)
    
    P = (M.T).dot(M)
    
    #calculate smoothing penalty
    K, coeff = spline_penalty(m, alphas[0])
    
    #add smoothing penalty to objective
    Ps = [matrix(P+alpha*K) for alpha in alphas]
    
    mu, s = np.mean(ys), np.std(ys)

    q = matrix(M.T.dot(-(ys-mu)/s))
    return Ps, q, G, h, m, M, K

def get_curve(x, y, n_gridpoints, alpha=0, min_x=None, max_x=None, init_vals=None, P=None, q=None, G=None, h=None, x_grid=None, isotonic=1, convexity=0):
    mu, s = np.mean(y), np.std(y)
    if P is None:
        P, q, G, h, x_grid, M, K = make_matrices(x, y, n_gridpoints, [alpha], min_x, max_x, isotonic=isotonic, convexity=convexity)
        P = P[0]
    if init_vals is None:
        ret = solvers.qp(P, q, G = G, h = h, init_vals=x_grid)
    else:
        ret = solvers.qp(P, q, G = G, h = h, init_vals=init_vals)
 
    fit_y = np.array(ret['x']).flatten()*s + mu
    x_grid = x_grid.flatten()
    
    x_expand = np.hstack((x_grid, x_grid.max() + 1, x_grid.max() + 2, x_grid.max() + 3, 1e14))
    slope = np.gradient(fit_y, x_grid)[-1]
    #print('success0')
    y_expand = np.hstack((fit_y, fit_y[-1] + slope, fit_y[-1] + 2 * slope, fit_y[-1] + 3 * slope, fit_y[-1] + 1e14 * slope))
    #print('success1')
    fx = interpolate.interp1d(x_expand, y_expand)
    #print('success2')
    dfdx = interpolate.interp1d(x_expand, np.gradient(y_expand, x_expand))
    #print('success3')
    df2dx2 = interpolate.interp1d(x_expand, np.gradient(np.gradient(y_expand, x_expand), x_expand))
    #print('success4')
    return x_grid, fit_y, fx, dfdx, df2dx2

def scan_alphas(xs, ys, n_gridpoints, alphas, isotonic=1, convexity=0, min_x = None, max_x=None):
    BICs = []
    Alinear = np.array([np.ones(len(xs)), xs])
    mu, s = np.mean(ys), np.std(ys)

    logH = np.linalg.slogdet(Alinear.dot(Alinear.T))[1] * (1./n_gridpoints)
    
    fit_fx = None
    Ps, q, G, h, x_grid, M, K = make_matrices(xs, ys, n_gridpoints, alphas, min_x, max_x, isotonic, convexity)
    dx = x_grid[1] - x_grid[0]
    maxBIC = -np.inf
    for alpha, P in zip(alphas[::-1], Ps[::-1]):
        print(f'Fitting standard with smoothing penalty:{alpha}                   ',end="\r")
        try:
            x_grid, fit_y, fx, dfdx, df2dx2 = get_curve(xs, ys, n_gridpoints, min_x = min_x, max_x=max_x, init_vals=fit_fx, P=P, q=q, G=G, h=h, x_grid=x_grid)
            LL = np.array(P).dot((fit_y-mu)/s).dot((fit_y-mu)/s) - 2 * np.array(q).flatten().dot((fit_y-mu)/s) #+ ys.dot(ys)
            S = np.linalg.pinv(np.array(P)) + np.diag(np.ones(n_gridpoints) * np.exp(-logH)) 
            BICs = [ 2 * LL - np.linalg.slogdet(S)[1] ] + BICs
            
            if BICs[0] > maxBIC:
                maxBIC = BICs[0]
                best_x_grid, best_fit_y, best_fx, bestdfdx, best_df2dx = x_grid, fit_y, fx, dfdx, df2dx2
        
        except:
            print(f'fit failed at alpha={alpha}', end='\r')
            BICs = [ -np.inf] + BICs
    
    return BICs, best_x_grid, best_fit_y, best_fx, bestdfdx, best_df2dx
