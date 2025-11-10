
"""

Выполнение семи шагов задания с Open3D.

"""
import os
import sys
import numpy as np
import open3d as o3d

INPUT_PATH = "uploads_files_2787791_Mercedes+Benz+GLS+580.obj"
# INPUT_PATH = "/Users/aleksandrsudro/Desktop/mercedes/uploads_files_2787791_Mercedes+Benz+GLS+580.obj" 
VOXEL_SIZE = 0.05 
POISSON_DEPTH = 7 
GRADIENT_AXIS = 'z' 
# --------------------------------
if not os.path.exists(INPUT_PATH):
    print(f"Ошибка: файл '{INPUT_PATH}' не найден. Положите модель в ту же папку и обновите INPUT_PATH.")
    sys.exit(1)
def print_mesh_info(mesh, label="Mesh"):
    v = np.asarray(mesh.vertices).shape[0]
    t = np.asarray(mesh.triangles).shape[0] if mesh.has_triangles() else 0
    has_colors = mesh.has_vertex_colors()
    has_normals = mesh.has_vertex_normals()
    has_intersections = mesh.is_self_intersecting()
    print(f"\n--- {label} ---")
    print(f"Вершин: {v}")
    print(f"Треугольников: {t}")
    print(f"Наличие цвета (вершины): {has_colors}")
    print(f"Наличие нормалей (вершины): {has_normals}")
    print(f"Наличие пересечений: {has_intersections}")
def print_pcd_info(pcd, label="PointCloud"):
    n = np.asarray(pcd.points).shape[0]
    has_colors = pcd.has_colors()
    has_normals = pcd.has_normals()
    print(f"\n--- {label} ---")
    print(f"Количество точек (вершин): {n}")
    print(f"Наличие цвета: {has_colors}")
    print(f"Наличие нормалей: {has_normals}")
   
    print(f"Наличие пересечений: Не применимо")
def print_voxel_info(voxel_grid, label="VoxelGrid"):
    voxels_count = len(voxel_grid.get_voxels())
    print(f"\n--- {label} ---")
    print(f"Количество вершин: {voxels_count}") 
    print(f"Наличие цвета: {False}")
    print(f"Наличие пересечений: Не применимо")
def wait_and_continue():
    print("Закройте окно визуализации, чтобы продолжить...")
    pass
# ------------------- ШАГ 1: Загрузка и визуализация -------------------
print("ШАГ 1: Попытка загрузить исходную модель как TriangleMesh...")
mesh = o3d.io.read_triangle_mesh(INPUT_PATH)
if mesh.is_empty():
    print("Ошибка: не удалось загрузить модель как TriangleMesh.")
    sys.exit(1)
if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()
print_mesh_info(mesh, "Исходный Mesh")
o3d.visualization.draw_geometries([mesh], window_name="Step 1: Исходная модель (Mesh)")
wait_and_continue()
print("Что я понял: Загрузка 3D-модели позволяет получить доступ к её геометрии в формате TriangleMesh, включая вершины, треугольники, цвета и нормали. Это базовый шаг для дальнейшей обработки.")
# ------------------- ШАГ 2: Преобразование в облако точек -------------------
print("\nШАГ 2: Преобразование в PointCloud...")
N_SAMPLE = 300000
pcd_from_mesh = mesh.sample_points_poisson_disk(number_of_points=min(N_SAMPLE, max(1000, len(mesh.vertices)*2)))
tmp_ply = "temp_pcd_for_assignment.ply"
o3d.io.write_point_cloud(tmp_ply, pcd_from_mesh)
pcd = o3d.io.read_point_cloud(tmp_ply)
print_pcd_info(pcd, "PCD после чтения через o3d.io.read_point_cloud()")
o3d.visualization.draw_geometries([pcd], window_name="Step 2: Point Cloud (прочитанный через read_point_cloud)")
wait_and_continue()
print("Что я понял: Преобразование meshes в point cloud упрощает некоторые операции, такие как вокселизация или реконструкция. Чтение через read_point_cloud демонстрирует загрузку точек из файла, и здесь мы видим потерю треугольников, но сохранение позиций точек.")
# ------------------- ШАГ 3: Реконструкция поверхности (Poisson) -------------------
print("\nШАГ 3: Реконструкция поверхности (Poisson)...")

pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
pcd = pcd.remove_non_finite_points() 
print("После downsample: точек =", len(pcd.points))
if not pcd.has_normals():
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE*2, max_nn=30))
# Ориентируем нормали consistently
pcd.orient_normals_consistent_tangent_plane(k=100)
mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=POISSON_DEPTH)
# Очистка по density сначала
densities = np.asarray(densities)
density_thresh = np.quantile(densities, 0.01)
verts_to_keep = np.where(densities > density_thresh)[0]
mesh_poisson = mesh_poisson.select_by_index(verts_to_keep)
# Затем crop по bounding box
bbox = pcd.get_axis_aligned_bounding_box()
mesh_poisson_clean = mesh_poisson.crop(bbox)
mesh_poisson_clean.remove_unreferenced_vertices()
mesh_poisson_clean.compute_vertex_normals()
print_mesh_info(mesh_poisson_clean, "Mesh после Poisson и crop")
o3d.visualization.draw_geometries([mesh_poisson_clean], window_name="Step 3: Реконструированный Mesh (Poisson)")
wait_and_continue()
print("Что я понял: Реконструкция Poisson позволяет восстановить поверхность из облака точек, создавая водонепроницаемый mesh. Crop удаляет артефакты, улучшая качество, но модель может потерять цвета, если они не были в PCD.")
# ------------------- ШАГ 4: Вокселизация -------------------
print("\nШАГ 4: Вокселизация (создание VoxelGrid из point cloud)...")
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=VOXEL_SIZE)
print_voxel_info(voxel_grid, "VoxelGrid")
o3d.visualization.draw_geometries([voxel_grid], window_name=f"Step 4: VoxelGrid (voxel_size={VOXEL_SIZE})")
wait_and_continue()
print("Что я понял: Вокселизация дискретизирует пространство в кубики, упрощая модель для анализа объёма. Количество 'вершин' здесь — это количество занятых вокселей, и цвета обычно не сохраняются.")
# ------------------- ШАГ 5: Добавление плоскости -------------------
print("\nШАГ 5: Добавление плоскости в сцену...")
# Динамическое позиционирование: горизонтальная плоскость посередине объекта по Z
bbox = mesh_poisson_clean.get_axis_aligned_bounding_box()
min_bound = bbox.min_bound
max_bound = bbox.max_bound
plane_size = max(max_bound[0] - min_bound[0], max_bound[1] - min_bound[1]) * 1.5
plane_z = (min_bound[2] + max_bound[2]) / 2
plane = o3d.geometry.TriangleMesh.create_box(width=plane_size, height=plane_size, depth=0.01)
translate_amount = [
    (min_bound[0] + max_bound[0]) / 2 - plane_size / 2,
    (min_bound[1] + max_bound[1]) / 2 - plane_size / 2,
    plane_z - 0.005
]
plane.translate(translate_amount)
plane.compute_vertex_normals()
plane.paint_uniform_color([0.8, 0.8, 0.8])
o3d.visualization.draw_geometries([mesh_poisson_clean, plane], window_name="Step 5: Объект + Плоскость (plane)")
wait_and_continue()
print("Что я понял: Добавление плоскости позволяет симулировать окружение или основу для объекта, что полезно для визуализации сцен или подготовки к обрезке.")
# ------------------- ШАГ 6: Обрезка по поверхности (клиппинг) -------------------
print("\nШАГ 6: Клиппинг: удаление точек по одну сторону от плоскости...")
plane_center = plane.get_center()
plane_normal = np.array([0.0, 0.0, 1.0]) # Нормаль вверх для горизонтальной плоскости
print(f"Параметры плоскости: центр={plane_center}, нормаль={plane_normal}")
points = np.asarray(mesh_poisson_clean.vertices)
dot_vals = np.dot(points - plane_center, plane_normal)
indices_keep = np.where(dot_vals >= 0)[0] # Удаляем точки ниже плоскости, показываем верхнюю часть
clipped_mesh = mesh_poisson_clean.select_by_index(indices_keep)
clipped_mesh.remove_unreferenced_vertices()
clipped_mesh.compute_vertex_normals()
print_mesh_info(clipped_mesh, "Mesh после клиппинга")
o3d.visualization.draw_geometries([clipped_mesh], window_name="Step 6: Mesh после клиппинга")
wait_and_continue()
print("Что я понял: Клиппинг по плоскости позволяет обрезать модель, удаляя ненужные части на основе математического условия (скалярное произведение), что сохраняет структуру mesh, но уменьшает количество элементов.")
# ------------------- ШАГ 7: Цвет и экстремумы -------------------
print("\nШАГ 7: Работа с цветом и выделение экстремумов...")
work_mesh = clipped_mesh if len(clipped_mesh.vertices) > 0 else mesh_poisson_clean
points = np.asarray(work_mesh.vertices)
work_mesh.vertex_colors = o3d.utility.Vector3dVector(np.zeros_like(points)) # Убираем цвета
ax = {'x':0, 'y':1, 'z':2}[GRADIENT_AXIS.lower()]
vals = points[:, ax]
min_idx = np.argmin(vals)
max_idx = np.argmax(vals)
min_point = points[min_idx]
max_point = points[max_idx]
vmin, vmax = vals.min(), vals.max()
if np.isclose(vmax, vmin):
    normalized = np.zeros_like(vals)
else:
    normalized = (vals - vmin) / (vmax - vmin)
colors = np.column_stack([normalized, np.zeros_like(normalized), 1 - normalized]) # Blue to red
work_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([work_mesh], window_name="Step 7: Mesh с градиентом")
print(f"Экстремумы по оси '{GRADIENT_AXIS}':")
print(f"Минимум (координаты): {min_point}")
print(f"Максимум (координаты): {max_point}")
# Маркеры (сферы) для экстремумов
sphere_radius = VOXEL_SIZE * 2 # Адаптивный размер
sphere_min = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
sphere_min.translate(min_point)
sphere_min.paint_uniform_color([0, 0, 1]) # Синий
sphere_max = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
sphere_max.translate(max_point)
sphere_max.paint_uniform_color([1, 0, 0]) # Красный
o3d.visualization.draw_geometries([work_mesh, sphere_min, sphere_max], window_name="Step 7: Градиент и экстремумы")
wait_and_continue()
print("Что я понял: Присвоение градиента по оси визуализирует высоты или другие измерения через цвета, а выделение экстремумов помогает идентифицировать ключевые точки модели для анализа.")

print("Файлы: временный pcd:", tmp_ply)
# Очистка
os.remove(tmp_ply)