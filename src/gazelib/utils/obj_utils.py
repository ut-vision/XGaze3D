import numpy as np

def load_color(obj_path):
	colors = []
	with open(obj_path) as f:
		lines = f.readlines()
	for line in lines:
		if len(line.split()) == 0:
			continue
		if line.split()[0] == 'v':
			colors.append([float(v) for v in line.split()[4:7]])
	colors = np.vstack(colors).astype(np.float32)
	return colors

def load_obj(obj_path):
    vertices = []
    colors = []
    with open(obj_path) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
            colors.append([float(v) for v in line.split()[4:7]])
    vertices = np.vstack(vertices).astype(np.float32)
    colors = np.vstack(colors).astype(np.float32)
    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype(np.int32) - 1
            
    return vertices, faces, colors

def write_obj_with_colors(obj_name, vertices, triangles, colors):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
    '''
    triangles = triangles.copy()
    triangles += 1 # meshlab start with 1
    
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
        
    # write obj
    with open(obj_name, 'w') as f:
        
        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)
        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            # s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)

##########################################################

def load_obj_full(obj_path):
    vertices = []
    colors = []

    vn = []
    vt = []

    with open(obj_path) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
            colors.append([float(v) for v in line.split()[4:7]])
        if line.split()[0] == 'vn':
            vn.append([float(v) for v in line.split()[1:4]])
        if line.split()[0] == 'vt':
            vt.append([float(v) for v in line.split()[1:3]])
    vertices = np.vstack(vertices).astype(np.float32)
    colors = np.vstack(colors).astype(np.float32)
    vn = np.vstack(vn).astype(np.float32)
    vt = np.vstack(vt).astype(np.float32)
    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype(np.int32) - 1
            
    return vertices, vn, vt, faces, colors

def write_obj_with_full(obj_name, vertices, vn, vt, triangles, colors):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
    '''
    triangles = triangles.copy()
    triangles += 1 # meshlab start with 1
    
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
        
    # write obj
    with open(obj_name, 'w') as f:
        
        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)

        for i in range(vn.shape[0]):
            s = 'vn {} {} {}\n'.format(vn[i, 0], vn[i, 1], vn[i, 2])
            f.write(s)
        for i in range(vt.shape[0]):
            s = 'vt {} {}\n'.format(vt[i, 0], vt[i, 1])
            f.write(s)
        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            f.write(s)