import open3d as o3d
import numpy as np


def crop_mesh(mesh, x,moins=False):
    """ Crop a shere mesh at a given x-coordinate. """
    # Accept either an Open3D TriangleMesh or a trimesh.Trimesh-like object
    # Accept Open3D TriangleMesh or trimesh-like objects
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles, dtype=np.int64)
    else:
        vertices = np.asarray(getattr(mesh, 'vertices', mesh.vertices))
        faces = np.asarray(getattr(mesh, 'faces', getattr(mesh, 'triangles', [])), dtype=np.int64)

    if vertices.size == 0:
        raise ValueError("Mesh has no vertices")

    # Select vertices based on x coordinate
    if moins:
        mask = vertices[:, 0] <= x
    else:
        mask = vertices[:, 0] >= x
    valid_indices = np.where(mask)[0]

    if valid_indices.size == 0:
        print(f"Warning: cropping at x={x} removed all vertices; returning original mesh")
        return mesh

    # Map old indices to new contiguous indices
    index_mapping = -np.ones(len(vertices), dtype=int)
    index_mapping[valid_indices] = np.arange(valid_indices.size)

    # Filter vertices
    cropped_vertices = vertices[valid_indices]

    # Filter faces: keep only faces whose all vertex indices survive the crop
    cropped_faces = []
    for face in faces:
        face = np.asarray(face, dtype=int)
        if face.size < 3:
            continue
        mapped = index_mapping[face]
        if np.all(mapped >= 0):
            cropped_faces.append(mapped.tolist())

    if len(cropped_faces) == 0:
        print(f"Warning: cropping at x={x} removed all faces; returning original mesh")
        return mesh

    cropped_faces = np.asarray(cropped_faces, dtype=np.int32)

    # Create new mesh
    cropped_mesh = o3d.geometry.TriangleMesh()
    cropped_mesh.vertices = o3d.utility.Vector3dVector(cropped_vertices)
    cropped_mesh.triangles = o3d.utility.Vector3iVector(cropped_faces)
    cropped_mesh.compute_vertex_normals()

    return cropped_mesh

def crop_sphere_mesh(mesh, center, radius):
    """ Crop a mesh to keep only vertices within a sphere of given center and radius. """
    # Accept either an Open3D TriangleMesh or a trimesh.Trimesh-like object
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles, dtype=np.int64)
    else:
        vertices = np.asarray(getattr(mesh, 'vertices', mesh.vertices))
        faces = np.asarray(getattr(mesh, 'faces', getattr(mesh, 'triangles', [])), dtype=np.int64)

    if vertices.size == 0:
        raise ValueError("Mesh has no vertices")

    # Compute distances from center
    distances = np.linalg.norm(vertices - center, axis=1)
    mask = distances <= radius
    valid_indices = np.where(mask)[0]

    if valid_indices.size == 0:
        print(f"Warning: cropping with sphere at center={center}, radius={radius} removed all vertices; returning original mesh")
        return mesh

    # Map old indices to new contiguous indices
    index_mapping = -np.ones(len(vertices), dtype=int)
    index_mapping[valid_indices] = np.arange(valid_indices.size)

    # Filter vertices
    cropped_vertices = vertices[valid_indices]

    # Filter faces: keep only faces whose all vertex indices survive the crop
    cropped_faces = []
    for face in faces:
        face = np.asarray(face, dtype=int)
        if face.size < 3:
            continue
        mapped = index_mapping[face]
        if np.all(mapped >= 0):
            cropped_faces.append(mapped.tolist())

    if len(cropped_faces) == 0:
        print(f"Warning: cropping with sphere at center={center}, radius={radius} removed all faces; returning original mesh")
        return mesh

    cropped_faces = np.asarray(cropped_faces, dtype=np.int32)

    # Create new mesh
    cropped_mesh = o3d.geometry.TriangleMesh()
    cropped_mesh.vertices = o3d.utility.Vector3dVector(cropped_vertices)
    cropped_mesh.triangles = o3d.utility.Vector3iVector(cropped_faces)
    cropped_mesh.compute_vertex_normals()

    return cropped_mesh