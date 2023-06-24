import numpy as np

def _projection_convex(y, b):
    sort= np.argsort(y.ravel()/b.ravel())
    y_hat = np.array(y).ravel()[sort]
    b_hat = np.array(b).ravel()[sort]
    nu = [(np.dot(y_hat[k:],b_hat[k:])-1)/np.dot(b_hat[k:], b_hat[k:]) for k in range(len(y_hat))]
    k = 0
    for i in range(len(nu)):
        if i == 0 :
            if nu[i]<=y_hat[i]/b_hat[i]:
                break
        elif (nu[i]>y_hat[i-1]/b_hat[i-1] and nu[i]<=y_hat[i]/b_hat[i]):
                k = i
                break
    return np.maximum(0, y-nu[k]*b)


def _projection_heuristic(y, b):
    y += b * ((((1-np.dot(np.transpose(b), y)) /
                (np.dot(np.transpose(b), b) + 1e-15))))
    y = np.maximum(0, y)
    y /= (np.dot(np.transpose(b), y) + 1e-15)
    return y


def gradient_descent(weight_param, metric,
                     lr=0.001, max_iter=100, verbose=1):
        
    objective = metric.eval()
    if verbose:
        print("GD : iter %i -- Obj %.4f"%(0, objective))
        
    for k in range(1, max_iter+1):
        grad = weight_param.gradient() @ metric.gradient()
        theta = weight_param.theta_ - lr * grad
        weight_param.update(theta)
        metric.update_weights(weight_param.predict("src"),
                              weight_param.predict("tgt"))
        objective = metric.eval()
        if verbose and k%10 == 0:
            print("GD : iter %i -- Obj %.4f"%(k, objective))
    return weight_param


def projected_gradient_descent(weight_param, metric, constraint_vect=None,
                               lr=0.001, projection="convex", max_iter=100, verbose=1):
    
    if projection == "convex":
        projection = _projection_convex
    else:
        projection = _projection_heuristic
    
    if constraint_vect is None:
        constraint_vect = np.ones(weight_param.theta_.shape[0])
        constraint_vect *= constraint_vect.sum()
    
    objective = metric.eval()
    if verbose:
        print("PGD : iter %i -- Obj %.4f"%(0, objective))
        
    for k in range(1, max_iter+1):
        grad = weight_param.gradient() @ metric.gradient()
        theta = weight_param.theta_ - lr * grad
        theta = projection(theta, 1/constraint_vect)
        weight_param.update(theta)
        metric.update_weights(weight_param.predict("src"), weight_param.predict("tgt"))
        objective = metric.eval()
        if verbose and k%10 == 0:
            print("PGD : iter %i -- Obj %.4f"%(k, objective))
    return weight_param


def frank_wolfe(weight_param, metric, constraint_vect=None, max_iter=100, verbose=1):
    
    if constraint_vect is None:
        constraint_vect = np.ones(weight_param.theta_.shape[0])
        constraint_vect *= constraint_vect.sum()
    
    objective = metric.eval()
    if verbose:
        print("Frank-Wolfe : iter %i -- Obj %.4f"%(0, objective))
    
    for k in range(1, max_iter+1):
        grad = weight_param.gradient() @ metric.gradient()
        index = np.argmin(grad * constraint_vect)
        vect = np.zeros(constraint_vect.shape[0])
        vect[index] = constraint_vect[index]
        lr = 2./(k+1.)
        weight_param.update((1 - lr) * weight_param.theta_ + lr * vect)
        metric.update_weights(weight_param.predict("src"), weight_param.predict("tgt"))
        objective = metric.eval()
        if verbose and k%10 == 0:
            print("Frank-Wolfe : iter %i -- Obj %.4f"%(k, objective))
    return weight_param