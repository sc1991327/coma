from facemesh import FaceData
from psbody.mesh import MeshViewers
import numpy as np
import argparse
import time
import readchar
from psbody.mesh import Mesh

parser = argparse.ArgumentParser()
parser.add_argument('--cnn', default='results/lips_up_predictions.npy', help='path to dataset')
parser.add_argument('--data', default='data/lips_up/', help='path to dataset')
parser.add_argument('--nz', type=int, default=8, help='size of the latent z vector')

opt = parser.parse_args()

reference_mesh_file = 'data/template.obj'
reference_changed_mesh_file = 'data/template_c.obj'
facedata = FaceData(nVal=1, train_file=opt.data+'train.npy', test_file=opt.data+'test.npy',
                    reference_mesh_file=reference_mesh_file, pca_n_comp=int(opt.nz), fitpca=True)

mean = np.mean(Mesh(filename=reference_mesh_file).v, axis=0)
mean_c = np.mean(Mesh(filename=reference_changed_mesh_file).v, axis=0)
std = np.std(Mesh(filename=reference_mesh_file).v, axis=0)
std_c = np.std(Mesh(filename=reference_changed_mesh_file).v, axis=0)

nTest = facedata.vertices_test.shape[0]
iTest = np.random.randint(0, nTest-1)

mean_vertices_oo = (np.reshape(Mesh(filename=reference_mesh_file).v, (facedata.n_vertex, 3)) * std) + mean
mean_vertices_oc = (np.reshape(Mesh(filename=reference_mesh_file).v, (facedata.n_vertex, 3)) * std_c) + mean_c
mean_vertices_co = (np.reshape(Mesh(filename=reference_changed_mesh_file).v, (facedata.n_vertex, 3)) * std) + mean
mean_vertices_cc = (np.reshape(Mesh(filename=reference_changed_mesh_file).v, (facedata.n_vertex, 3)) * std_c) + mean_c

test_vertices = (np.reshape(facedata.vertices_test[iTest], (facedata.n_vertex, 3)) * std) + mean
test_vertices_c = (np.reshape(facedata.vertices_test[iTest], (facedata.n_vertex, 3)) * std_c) + mean_c

pca_outputs = facedata.pca.inverse_transform(facedata.pca.transform(np.reshape(facedata.vertices_test[iTest], (1, facedata.n_vertex*3))))
pca_vertices = (np.reshape(pca_outputs, (facedata.n_vertex, 3)) * std) + mean
pca_vertices_c = (np.reshape(pca_outputs, (facedata.n_vertex, 3)) * std_c) + mean_c

outmesh = np.zeros((4, facedata.n_vertex*3))
outmesh[0] = np.reshape(test_vertices, (facedata.n_vertex*3,))
outmesh[1] = np.reshape(test_vertices_c, (facedata.n_vertex*3,))
outmesh[2] = np.reshape(pca_vertices, (facedata.n_vertex*3,))
outmesh[3] = np.reshape(pca_vertices_c, (facedata.n_vertex*3,))

# vertices = outmesh[0].reshape((facedata.n_vertex, 3))
# mesh = Mesh(v=vertices, f=facedata.reference_mesh.f)
# mesh.write_obj(opt.data + '/' + str(iTest) + '_outmesh_template.obj')

viewer = MeshViewers(window_width=800, window_height=200, shape=[1, 4], titlebar='Meshes')
while(1):
    facedata.show_mesh(viewer=viewer, mesh_vecs=outmesh, figsize=(1, 4))
    inp = readchar.readchar()
    if inp == "e":
        break
