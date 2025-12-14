
import os
import sys
from src import transformation
from src.loadfbx import loadfbx
from src.transformation import Transformation
from src.utils import *


if __name__ == "__main__":
    
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
        
    # Load FBX meshes
    source_loader = loadfbx(source_path)
    target_loader = loadfbx(target_path)
    source_mesh_origin = source_loader.load_fbx_mesh()
    target_mesh_origin = target_loader.load_fbx_mesh()


    ### crop meshes
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
    source_center = rotation[:3, :3] @ source_center + translation[:3, 3]
    target_center = target_mesh_origin.get_center()
    R = 0.08
    ### crop les mesh d'une sphère de rayon R
    commune_source = crop_sphere_mesh(source_mesh, source_center, R)
    commune_target = crop_sphere_mesh(target_mesh, target_center, R)


    # Perform robust registration
    transformer = Transformation(commune_source, commune_target)
    transformed_source_mesh, transformation_matrix, fitness, inlier_rmse = transformer.robust_registration(voxel_size=0.003, use_ransac=True)

    ### transform the source mesh cropped
    source_mesh.transform(transformation_matrix)

    #### fusion clothest point entre les deux mesh après registration ####
    mesh_combined = source_mesh + target_mesh
    mesh_combined = mesh_combined.merge_close_vertices(0.00002)


    # Visualize the result
    target_mesh.paint_uniform_color([0, 1, 0])  # Green
    source_mesh.paint_uniform_color([0, 0, 1])  # Blue

    # Visualize the meshes
    try:
        o3d.visualization.draw_geometries([target_mesh, source_mesh])
        o3d.visualization.draw_geometries([mesh_combined], window_name='Combined Mesh After Registration')
    except Exception:
        print("Visualization failed or not available in this environment. Registration finished.")

