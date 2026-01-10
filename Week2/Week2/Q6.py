import torch as T

x = T.tensor(1.0)
y = T.tensor(1.0, requires_grad=True)
z = T.tensor(1.0)

vs = {'a': 2 * x} # a
vs['b'] = T.sin(y)
vs['c'] = vs['a'] / vs['b']
vs['d'] = vs['c'] * z
vs['e'] = T.log(vs['d'] + 1)
vs['f'] = T.tanh(vs['e'])

for k in vs:
    print(f'{k} = {vs[k]}')

vs['f'].backward()
print(f'df/dy = {y.grad}')


# Manual
ds = {'df/de': 1 - T.tanh(vs['e']) ** 2}
ds['de/dd'] = 1 / (vs['d'] + 1)
ds['dd/dc'] = z
ds['dc/db'] = -vs['a'] / vs['b'] ** 2
ds['db/dy'] = T.cos(y)

print()
for k in ds:
    print(f'{k} = {ds[k]}')

final = ds['db/dy'] * ds['dc/db'] * ds['dd/dc'] * ds['de/dd'] * ds['df/de']
print(f'\ndf/dy = {final}')