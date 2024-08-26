import torch
import torch.nn.functional as F
import numpy as np 
import sys
sys.path.append('/autofs/space/durian_001/users/xh999/regre4regis/proj_supersynth_registration')
import ext.interpol

def weighted_mse(fixed, moving, weights):
    # return torch.sum(weights * (fixed - moving) ** 2) / torch.sum(weights)
    return torch.sum(weights[None,:] * (fixed - moving) ** 2) / torch.sum(weights)

def weighted_bspline_registration(fixed_image, moving_image, moving_weights, control_points, max_iterations=100, learning_rate=1.0):

    # optimizer = torch.optim.Adam([control_points], lr=learning_rate)
    optimizer = torch.optim.LBFGS([control_points], lr=learning_rate, line_search_fn='strong_wolfe')
    for iteration in range(max_iterations):
        print('Iteration:', iteration)
        def closure():
            optimizer.zero_grad()
            loss = weighted_mse(fixed_image, ext.interpol.resize(control_points, shape = moving_image.shape, interpolation=3, prefilter=True), moving_weights)
            loss.backward()
            print('Iteration:', iteration, 'Loss:', loss)
            return loss
        optimizer.step(closure)

        if control_points.grad.norm() < 1e-5:
            print("Convergence reached.")
            break
        
    # Apply the final transformation
    final_registered_image = ext.interpol.resize(control_points, shape = moving_image.shape, interpolation=3, prefilter=True)

    return final_registered_image, control_points.detach()

# def unrolled_gradient_descent(control_points, hyperparams, num_steps, learning_rate):
#     params = control_points

#     # Unroll the gradient descent for num_steps iterations
#     for step in range(num_steps):
#         loss = weighted_mse(fixed_image, ext.interpol.resize(control_points, shape = moving_image.shape, interpolation=3, prefilter=True), moving_weights)
#         grads = compute_gradients(loss, params)
#         params = params - learning_rate * grads

#     # After unrolling, compute a higher-level objective
#     final_loss = compute_higher_level_loss(params)
    
#     # Compute gradients w.r.t initial params or hyperparameters
#     final_grads = compute_gradients(final_loss, initial_params or hyperparams)

#     return final_grads

# Example Usage
if __name__ == "__main__":
    # Example data (replace these with actual medical images and weights)
    fixed_image = torch.rand(3, 256, 256, 256).cuda()  # Fixed image with shape (D, H, W)
    moving_image = torch.rand((3, 256, 256, 256), requires_grad=True).cuda()  # Moving image with shape (D, H, W)
    moving_weights = torch.ones(256, 256, 256).cuda()  # Uniform weights with shape (D, H, W)

    control_points = torch.zeros(3, 20, 20, 20, device=moving_image.device, requires_grad=True) # Initial control points with shape (Cx, Cy, Cz, 3)

    # control_points_shape = (8, 8, 8)  # Control point grid shape (Cx, Cy, Cz)
    # control_points_shape = tuple(np.ceil(np.array(moving_image.shape[1:]) / 5).astype(int))

    # Perform registration
    registered_image, final_control_points = weighted_bspline_registration(
        fixed_image, moving_image, moving_weights, control_points
    )

    print("Registration complete.")
    print("Final control points:", final_control_points.shape)

    