import torch


def frank_wolfe_solver(gradients, iterations):
    # TODO shape gradients into nxm tensor
    num_tasks, num_params = gradients.size()
    alphas = torch.ones(num_tasks) / num_tasks
    M = gradients @ gradients.t()
    for _ in range(iterations):
        t = torch.argmin(M @ alphas.view(-1, 1), dim=0)

        # Algorithm 1
        theta1 = alphas @ gradients
        theta2 = gradients[t].view(-1)
        a = theta1 @ theta1
        b = theta1 @ theta2
        c = theta2 @ theta2
        if b >= a:
            gamma = 0
        elif b >= c:
            gamma = 1
        else:
            gamma = theta1 @ (theta1 - theta2) / (
                (theta2 - theta1) @ (theta2 - theta1))
        alphas = (1 - gamma) * alphas
        alphas[t] += gamma

    return alphas


if __name__ == '__main__':
    gradients = torch.tensor([[-4, 1], [2, 1]]).float()
    alphas = frank_wolfe_solver(gradients, 3)
    print(alphas @ gradients)
