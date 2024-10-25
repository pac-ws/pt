import torch

robot_id = 2

# Load all_maps.pt
all_maps = torch.load('all_maps.pt').to('cpu')
print(f'shape of all_maps: {all_maps.shape}')

# Load cnn_input.pt, which is of type script_module
cnn_input = list(torch.load('cnn_input.pt').parameters())[0][0].to('cpu')


# Check equality
map = all_maps[robot_id]
print(f'All equal: {torch.all(torch.eq(map, cnn_input))}')

# Check difference
print(f'Max diff: {torch.abs(map - cnn_input).max()}')

# Check close
print(f'All close: {torch.allclose(map, cnn_input)}')

print(map.shape)
print(cnn_input.shape)

# Check each channel
for i in range(0, 4):
    print(torch.abs(map[i] - cnn_input[i]).max())
    print(f'Channel {i} sum: {torch.sum(map[i])} {torch.sum(cnn_input[i])}')
    print(f'Channel {i} all equal: {torch.all(torch.eq(map[i], cnn_input[i]))}')




print('-------------------')
all_cnn_output = torch.load('all_cnn_output.pt').to('cpu')
cnn1 = all_cnn_output[robot_id]

cnn2 = list(torch.load('cnn_output.pt').parameters())[0].to('cpu')[0]

print(f'cnn1 shape: {cnn1.shape}')
print(f'cnn2 shape: {cnn2.shape}')
print(f'all equal: {torch.all(torch.eq(cnn1, cnn2))}')
print(f'all close: {torch.allclose(cnn1, cnn2)}')
print(f'max diff: {torch.abs(cnn1 - cnn2).max()}')
print(f'relative diff: {(torch.abs(cnn1 - cnn2)/torch.abs(cnn2)).max()}')
print(f'cnn1 max: {torch.abs(cnn1).max()}')
print(f'cnn1 min: {torch.abs(cnn1).min()}')
print(f'cnn1: {cnn1}')

