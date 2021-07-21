import torch
import math

dim = 1000
features = 784
basis = torch.randn(dim, features)
base = torch.empty(dim).uniform_(0.0, 2*math.pi)

printing = False

def temp(x: torch.Tensor):
    '''
    Encodes each data point in `x` to high dimensional space.
    The encoded representation of the `(n?, features)` samples described
    in :math:`x`, is the `(n?, dim)` matrix :math:`H`:

    .. math:: H_{ij} = \cos(x_i \cdot B_j + b_j) \sin(x_i \cdot B_j)

    Note:
        This encoder is very sensitive to data preprocessing. Try
        making input have unit norm (normalizing) or standarizing each
        feature to have mean=0 and std=1/sqrt(features) (scaling).

    Args:
        x (:class:`torch.Tensor`): The original data points to encode. Must
            have size `(n?, features)`.

    Returns:
        :class:`torch.Tensor`: The high dimensional representation of each
        of the `n?` data points in x, which respects the equation given
        above. It has size `(n?, dim)`.
    '''

    n = x.size(0)
    bsize = math.ceil(0.01 * n)
    h = torch.empty(n, dim, device=x.device, dtype=x.dtype)
    temp = torch.empty(bsize, dim, device=x.device, dtype=x.dtype)
    if printing:
        print('basis.T', basis.T)
        print('n', n)
        print('h', h, h.sum(), h.size())
        print('bsize', bsize)
        print('temp', temp, temp.sum(), temp.size())

    # we need batches to remove memory usage
    for i in range(0, n, bsize):
        # torch.matmul(x[i:i+bsize], self.basis.T, out=temp)
        # torch.add(temp, self.base, out=h[i:i+bsize])
        # h[i:i+bsize].cos_().mul_(temp.sin_())
        if printing:
            print('for0 x[i:i+bsize]', x[i:i + bsize])
            print('for0 temp', temp)
            print('for0 h[i:i+bsize]', h[i:i + bsize])
        torch.matmul(x[i:i + bsize], basis.T, out=temp)
        if printing:
            print('for1 x[i:i+bsize]', x[i:i + bsize])
            print('for1 temp', temp)
            print('for1 h[i:i+bsize]', h[i:i + bsize])
        torch.add(temp, base, out=h[i:i + bsize])
        if printing:
            print('for2 self.base', base)
            print('for2 x[i:i+bsize]', x[i:i + bsize])
            print('for2 temp', temp)
            print('for2 h[i:i+bsize]', h[i:i + bsize])
        h[i:i + bsize].mul_(temp)  # element-wise

        if printing:
            print('for3 x[i:i+bsize]', x[i:i + bsize])
            print('for3 temp', temp)
            print('for3 h[i:i+bsize]', h[i:i + bsize])

        x_back = torch.empty(bsize, 784, device=x.device, dtype=x.dtype)
        # print(x[i:i + bsize].size(), x_back.size())
        # print(temp.size(), basis.T.size(), basis.T.pinverse().size())
        # torch.matmul(temp, basis.T.pinverse(), out=x_back)
        # print(x[i:i + bsize])
        # print(x_back)

        # x dot basis.T = temp
        # (temp + base) * temp = h
        # temp^2 + base * temp = h
        # temp = (-base +- (base^2 - 4*(-h)) ** 2 / 2

        # print(base.size())
        # print(-1 * base)
        # exit()

        # print(temp.mul(temp) + temp.mul(base))
        # print(h[i:i + bsize])
        # print(base[:10])
        # print(base.mul(base)[:10])

        a = -1 * base
        b = base.mul(base).add(h[i:i + bsize].mul(4)).sqrt()
        c1 = (a + b).div(2)
        c2 = (a - b).div(2)

        print('temp', temp)
        print('c1', c1)
        print('c2', c2)

        print('x[i:i + bsize]', x[i:i + bsize])
        print('temp', torch.matmul(temp, basis.T.pinverse()))
        print('c1', torch.matmul(c1, basis.T.pinverse()))
        print('c2', torch.matmul(c2, basis.T.pinverse()))

        # print(a.size())
        # print(b.size())
        # print(c1.size())
        # print(c2.size())
        # print(x[i:i + bsize].size())
        # print(temp.size())
        # print(a[:5])
        # print(b[:, :5])
        # print(c1[:, :5])

        print()

# x = torch.rand([4, 4])
# print(x)
# print(x.inverse())
# print(x.pinverse())
# print(torch.linalg.inv(x))
# print(x.T)
# print(torch.matmul(x, x.inverse()))

temp(torch.rand([1000, 784]))

# a = torch.rand([4, 4])
# b = torch.rand([4, 4])
# print(a)
# print(b)
# a.mul_(b)
# print(a)