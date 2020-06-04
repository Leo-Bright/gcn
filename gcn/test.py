import sys
import numpy as np
import json

m = np.array([[1,1,1,1],
          [2,2,2,2],
          [3,3,3,3],
          [4,4,4,4]])

print(m)
print("======")
m[[2,1],:] = m[[1,2],:]
print(m)

# with open('sanfrancisco/osm_data/sf_segments_tiger_nametype.json') as f:
#     seg_tag_dict = json.loads(f.readline())
#     tag_count = {}
#     for key in seg_tag_dict:
#         tag = seg_tag_dict[key]
#         if tag not in tag_count:
#             tag_count[tag] = 0
#         tag_count[tag] += 1
#
#     for key in tag_count:
#         print(key, tag_count[key])

with open('sanfrancisco/sanfrancisco_nodes_with_all.tag') as f:

    tag_count = {}
    nodes = set()
    result = {}

    for line in f:
        osmid_tag = line.strip().split(' ')
        osmid = osmid_tag[0]
        tag = osmid_tag[1]
        if tag == '3':
            result[osmid] = {'highway': 'turning_circle'}
        if osmid in nodes:
            continue
        nodes.add(osmid)
        if tag not in tag_count:
            tag_count[tag] = 0
        tag_count[tag] += 1

    for key in tag_count:
        print(key, tag_count[key])

    # with open('sanfrancisco/osm_data/nodes_turning_circle.json', 'w+') as f:
    #     json.dump(result, f)

    circle_count = 0
    for key in result:
        circle_count += 1
    print(circle_count)