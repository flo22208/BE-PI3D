import open3d as o3d
import numpy as np
from copy import deepcopy

class Transformation:
    def __init__(self, meshA,meshB,):
        self.meshA = meshA
        self.meshB = meshB

    def compute_fpfh_features(self, pcd, voxel_size):
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

    def ransac_based_registration(self, source_pcd, target_pcd, voxel_size):
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
        source_fpfh = self.compute_fpfh_features(source_pcd, voxel_size)
        target_fpfh = self.compute_fpfh_features(target_pcd, voxel_size)
        
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

    def refine_registration_icp(self, source_pcd, target_pcd, initial_transform, voxel_size):
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

    def robust_registration(self,voxel_size=0.005, use_ransac=True):
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
        source_pcd = self.meshA.sample_points_uniformly(number_of_points=10000)
        target_pcd = self.meshB.sample_points_uniformly(number_of_points=10000)
        
        # Downsampling pour accélérer les calculs
        source_down = source_pcd.voxel_down_sample(voxel_size)
        target_down = target_pcd.voxel_down_sample(voxel_size)
        
        print(f"Source points: {len(source_pcd.points)} -> {len(source_down.points)} after downsampling")
        print(f"Target points: {len(target_pcd.points)} -> {len(target_down.points)} after downsampling")
        
        if use_ransac:
            # Étape 1: RANSAC avec feature matching pour alignement grossier robuste
            result_ransac = self.ransac_based_registration(source_down, target_down, voxel_size)
            
            # Étape 2: Raffiner avec ICP point-to-plane
            result_icp = self.refine_registration_icp(source_pcd, target_pcd, 
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
        transformed_source_mesh = deepcopy(self.meshA)
        transformed_source_mesh.transform(transformation)
        
        return transformed_source_mesh, transformation, fitness, inlier_rmse

    def icp_registration(self, commune_source_mesh,commune_target_mesh,source_mesh, target_mesh, threshold=0.02, transformation_init=np.eye(4), max_iteration=50):
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