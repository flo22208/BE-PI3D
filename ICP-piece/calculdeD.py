##### à partir de deux mesh fais la correspondance par ICP #####
import numpy as np
import trimesh
import open3d as o3d
from copy import deepcopy
from fbxloader import FBXLoader
import os, sys
from scipy.spatial import cKDTree

def compute_fpfh_features(pcd, voxel_size):
    """
    Calcule les descripteurs FPFH (Fast Point Feature Histograms) pour un nuage de points.
    FPFH capture la géométrie locale autour de chaque point.
    """
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh

def ransac_based_registration(source_pcd, target_pcd, voxel_size):
    """
    Alignement robuste basé sur RANSAC avec correspondances de features FPFH.
    
    Étapes:
    1. Calcul des features FPFH pour source et target
    2. Matching des features pour trouver des correspondances
    3. RANSAC pour éliminer les outliers et trouver une transformation robuste
    
    Cette méthode est robuste même avec un mauvais alignement initial.
    """
    print("\n=== RANSAC-based Registration ===")
    
    # 1. Calculer les features FPFH
    print("Computing FPFH features...")
    source_fpfh = compute_fpfh_features(source_pcd, voxel_size)
    target_fpfh = compute_fpfh_features(target_pcd, voxel_size)
    
    # 2. RANSAC registration basée sur les correspondances de features
    distance_threshold = voxel_size * 1.5
    print(f"RANSAC with distance threshold: {distance_threshold}")
    
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd, target_pcd, source_fpfh, target_fpfh,
        mutual_filter=True,  # Filtrage mutuel pour de meilleures correspondances
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,  # Nombre de points pour estimer la transformation
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    
    print(f"RANSAC fitness: {result_ransac.fitness}")
    print(f"RANSAC inlier RMSE: {result_ransac.inlier_rmse}")
    
    return result_ransac

def refine_registration_icp(source_pcd, target_pcd, initial_transform, voxel_size):
    """
    Raffine l'alignement avec ICP point-to-plane après RANSAC.
    Point-to-plane ICP est plus précis que point-to-point.
    """
    print("\n=== ICP Refinement ===")
    distance_threshold = voxel_size * 0.4
    print(f"ICP distance threshold: {distance_threshold}")
    
    result_icp = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, distance_threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    print(f"ICP fitness: {result_icp.fitness}")
    print(f"ICP inlier RMSE: {result_icp.inlier_rmse}")
    
    return result_icp

def robust_registration(commune_source_mesh, commune_target_mesh, source_mesh, target_mesh, 
                       voxel_size=0.005, use_ransac=True):
    """
    Méthode d'alignement robuste combinant RANSAC + Feature Matching + ICP.
    
    Arguments:
    - voxel_size: taille du voxel pour le downsampling et les calculs de features
    - use_ransac: si True, utilise RANSAC+FPFH, sinon utilise ICP classique
    
    Retourne:
    - transformed_source_mesh: mesh source transformé
    - transformation: matrice 4x4 de transformation
    - fitness: score de qualité [0,1]
    - inlier_rmse: erreur quadratique moyenne
    """
    # Convertir les meshes en nuages de points
    source_pcd = commune_source_mesh.sample_points_uniformly(number_of_points=10000)
    target_pcd = commune_target_mesh.sample_points_uniformly(number_of_points=10000)
    
    # Downsampling pour accélérer les calculs
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)
    
    print(f"Source points: {len(source_pcd.points)} -> {len(source_down.points)} after downsampling")
    print(f"Target points: {len(target_pcd.points)} -> {len(target_down.points)} after downsampling")
    
    if use_ransac:
        # Étape 1: RANSAC avec feature matching pour alignement grossier robuste
        result_ransac = ransac_based_registration(source_down, target_down, voxel_size)
        
        # Étape 2: Raffiner avec ICP point-to-plane
        result_icp = refine_registration_icp(source_pcd, target_pcd, 
                                            result_ransac.transformation, voxel_size)
        
        transformation = result_icp.transformation
        fitness = result_icp.fitness
        inlier_rmse = result_icp.inlier_rmse
    else:
        # ICP classique (méthode originale)
        print("\n=== Classic ICP ===")
        result_icp = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, 0.02, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        transformation = result_icp.transformation
        fitness = result_icp.fitness
        inlier_rmse = result_icp.inlier_rmse
    
    print("\n=== Final Transformation Matrix ===")
    print(transformation)
    
    # Appliquer la transformation au mesh complet
    transformed_source_mesh = deepcopy(source_mesh)
    transformed_source_mesh.transform(transformation)
    
    return transformed_source_mesh, transformation, fitness, inlier_rmse

def icp_registration(commune_source_mesh,commune_target_mesh,source_mesh, target_mesh, threshold=0.02, transformation_init=np.eye(4), max_iteration=50):
    # Convert meshes to point clouds
    source_pcd = commune_source_mesh.sample_points_uniformly(number_of_points=10000)
    target_pcd = commune_target_mesh.sample_points_uniformly(number_of_points=10000)

    # Perform ICP registration
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, transformation_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )

    ## print transformation matrix 
    print("Transformation matrix:")
    print(reg_p2p.transformation)

    # Apply the transformation to the source mesh
    transformed_source_mesh = deepcopy(source_mesh)
    transformed_source_mesh.transform(reg_p2p.transformation)

    return transformed_source_mesh, reg_p2p.transformation, reg_p2p.fitness, reg_p2p.inlier_rmse

### load mesh fbx ###
def load_fbx_mesh(file_path):
    fbx = FBXLoader(file_path) # load from file or bytes

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

# Example usage:
if __name__ == "__main__":
    # Build source and target paths relative to this script
    current_folder = os.getcwd()
    source_path = os.path.join(current_folder, 'silver-hemiobol-of-kyzikos', 'source', 'model', 'model.fbx')
    target_path = os.path.join(current_folder, 'silver-hemiobol-of-kyzikos', 'target', 'model', 'model.fbx')

    # Prefer explicit existence checks with helpful messages
    if not os.path.exists(source_path):
        print(f"Source FBX not found: {source_path}")
        sys.exit(1)
    if not os.path.exists(target_path):
        print(f"Target FBX not found: {target_path}\nIf you only have one file, set target_path = source_path or copy the file to the expected location.")
        sys.exit(1)

    source_mesh_origin = load_fbx_mesh(source_path)
    target_mesh_origin = load_fbx_mesh(target_path)

    ## crop les mesh 
    source_mesh = crop_mesh(source_mesh_origin, x=-0.1, moins=False)
    target_mesh = crop_mesh(target_mesh_origin, x=0.1, moins=True)

    ## déplacer et tourner le source mesh un peu a gauche pour eviter le chevauchement initial
    translation = np.array([[1, 0, 0, -0.1],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
    # très petite rotation autour de l'axe Z (10 degrés)
    theta = np.deg2rad(10)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s, 0, 0],
                         [s,  c, 0, 0],
                         [0,  0, 1, 0],
                         [0,  0, 0, 1]])
    source_mesh.transform(rotation)
    source_mesh.transform(translation)



    ### récupérer les centres des mesh
    source_center = source_mesh_origin.get_center()
    ### appliquer la tranlation au centre
    #source_center = rotation[:3, :3] @ source_center + translation[:3, 3]
    # trouver le source center après transformation
    d = 

    target_center = target_mesh_origin.get_center()
    R = 0.08
    ### crop les mesh d'une sphère de rayon R
    commune_source = crop_sphere_mesh(source_mesh, source_center, R)
    commune_target = crop_sphere_mesh(target_mesh, target_center, R)


    ## visualize les mesh avant registration sur deux fenetres
    try:
        #o3d.visualization.draw_geometries([source_mesh], window_name='Source Mesh Before Registration')
        #o3d.visualization.draw_geometries([target_mesh], window_name='Target Mesh Before Registration')
        #o3d.visualization.draw_geometries([commune_source], window_name='Cropped Source Mesh')
        #o3d.visualization.draw_geometries([commune_target], window_name='Cropped Target Mesh')
        pass
    except Exception:
        print("Visualization failed or not available in this environment. Proceeding to registration.")


    # Perform robust registration with RANSAC + FPFH + ICP
    print("\n" + "="*60)
    print("MÉTHODE ROBUSTE: RANSAC + Feature Matching + ICP")
    print("="*60)
    transformed_source_mesh, transformation, fitness, inlier_rmse = robust_registration(
        commune_source, commune_target, source_mesh, target_mesh, 
        voxel_size=0.003, use_ransac=True
    )
    
    # Décommenter pour comparer avec ICP classique:
    # print("\n" + "="*60)
    # print("MÉTHODE CLASSIQUE: ICP uniquement")
    # print("="*60)
    # transformed_source_mesh, transformation, fitness, inlier_rmse = icp_registration(
    #     commune_source, commune_target, source_mesh, target_mesh
    # )

    #### fusion clothest point entre les deux mesh après registration ####

    mesh_combined = transformed_source_mesh + target_mesh
    mesh_combined = mesh_combined.merge_close_vertices(0.00002)


    # Print results
    print("Transformation Matrix:\n", transformation)
    print("Fitness:", fitness)
    print("Inlier RMSE:", inlier_rmse)

    # Visualize the result
    #source_mesh.paint_uniform_color([1, 0, 0])  # Red
    target_mesh.paint_uniform_color([0, 1, 0])  # Green
    transformed_source_mesh.paint_uniform_color([0, 0, 1])  # Blue

    # Visualize the meshes
    try:
        o3d.visualization.draw_geometries([target_mesh, transformed_source_mesh])
        o3d.visualization.draw_geometries([mesh_combined], window_name='Combined Mesh After Registration')
    except Exception:
        print("Visualization failed or not available in this environment. Registration finished.")