from fbxloader import FBXLoader
import open3d as o3d
import numpy as np

class loadfbx:
    def __init__(self, filepath):
        self.filepath = filepath
        self.loader = FBXLoader(filepath)

    def load_fbx_mesh(self):
        fbx = FBXLoader(self.filepath) # load from file or bytes

        mesh = fbx.export_trimesh()

        # If the loader returned a trimesh.Trimesh, convert it to an Open3D TriangleMesh
        try:
            import trimesh
            if isinstance(mesh, trimesh.Trimesh):
                # export an obj copy for debugging/inspection
                try:
                    mesh.export('model.obj')
                except Exception:
                    pass

                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
                # ensure faces are integer dtype
                faces = np.asarray(mesh.faces, dtype=np.int32)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
                o3d_mesh.compute_vertex_normals()
                return o3d_mesh
        except Exception:
            # If trimesh is not available or conversion fails, fall through
            pass

        # If mesh is already an Open3D mesh-like object, try to return it directly
        try:
            return mesh
        except Exception:
            raise RuntimeError('Unsupported mesh type returned by FBXLoader')