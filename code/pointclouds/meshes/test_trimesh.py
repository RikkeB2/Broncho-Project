import trimesh

print(trimesh.__version__)
mesh = trimesh.creation.box()
print(mesh.is_manifold)