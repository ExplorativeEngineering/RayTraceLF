# each subimg is ulense square

ulenses = 8
sub_offs = []
for x in range(5):
    sub_offX = x * ulenses
    for y in range(5):
        sub_offY = y * ulenses
        sub_offs.append([sub_offX, sub_offY])
print(sub_offs)
