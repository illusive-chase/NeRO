from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type

from rfstudio.engine.task import Task, TaskGroup
from rfstudio.graphics import Cameras, Texture2D, TextureCubeMap, TriangleMesh, RGBAImages
from rfstudio.graphics.shaders import BaseShader, ShadingContext, NormalShader
from rfstudio.io import dump_float32_image
from rfstudio.data import RelightDataset
from rfstudio.visualization import TabularFigures
from rfstudio.ui import console

from jaxtyping import Float32
from torch import Tensor
import torch
import numpy as np

def srgb_to_linear(srgb) -> Tensor:
    if isinstance(srgb, torch.Tensor):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = torch.clamp(((200 * srgb + 11) / (211)), min=eps) ** (12 / 5)
        return torch.where(srgb <= 0.04045, linear0, linear1)
    else:
        raise NotImplementedError

@dataclass
class VertexShader(BaseShader[RGBAImages]):

    vertex_colors: Float32[Tensor, "V 3"] = ...

    def get_image_class(self) -> Type[RGBAImages]:
        return RGBAImages

    def shade(self, context: ShadingContext) -> Float32[Tensor, "1 H W 3"]:
        assert self.vertex_colors.shape == (context.mesh.num_vertices, 3)
        return context.vertex_attribute_map(self.vertex_colors)

@dataclass
class Tester(Task):

    mesh: Path = ...
    material: Path = ...
    output: Path = ...
    z_up: bool = True

    dataset: RelightDataset = RelightDataset(path=...)

    def run(self) -> None:
        assert (self.material / 'albedo.npy').exists()
        assert (self.material / 'metallic.npy').exists()
        assert (self.material / 'roughness.npy').exists()
        albedo = srgb_to_linear(torch.from_numpy(np.load(self.material / 'albedo.npy')).float()).to(self.device)
        metallic = srgb_to_linear(torch.from_numpy(np.load(self.material / 'metallic.npy')).float()).to(self.device).expand_as(albedo).contiguous()
        roughness = srgb_to_linear(torch.from_numpy(np.load(self.material / 'roughness.npy')).float()).to(self.device).expand_as(albedo).contiguous()

        (self.output / 'albedo').mkdir(parents=True, exist_ok=True)
        (self.output / 'metallic').mkdir(parents=True, exist_ok=True)
        (self.output / 'roughness').mkdir(parents=True, exist_ok=True)
        (self.output / 'normal').mkdir(parents=True, exist_ok=True)

        mesh = TriangleMesh.from_file(self.mesh).to(self.device)
        mesh.replace_(vertices=mesh.vertices * (4 / 3), indices=mesh.indices[:, [0, 2, 1]].contiguous())
        if not self.z_up:
            mesh.replace_(
                vertices=(
                    torch.tensor([
                        [0, -1, 0],
                        [0, 0, 1],
                        [-1, 0, 0],
                    ]).float().to(self.device) @ mesh.vertices.unsqueeze(-1)
                ).squeeze(-1),
            )

        self.dataset.to(self.device)
        cameras = self.dataset.get_inputs(split='test')[...]
        albedo_shader = VertexShader(vertex_colors=albedo)
        metallic_shader = VertexShader(vertex_colors=metallic)
        roughness_shader = VertexShader(vertex_colors=roughness)

        with console.progress(desc='Rendering') as ptrack:
            for i, camera in enumerate(ptrack(cameras)):
                dump_float32_image(self.output / 'albedo' / f'{i}.png', mesh.render(camera, shader=albedo_shader).item())
                dump_float32_image(self.output / 'metallic' / f'{i}.png', mesh.render(camera, shader=metallic_shader).item())
                dump_float32_image(self.output / 'roughness' / f'{i}.png', mesh.render(camera, shader=roughness_shader).item())
                dump_float32_image(self.output / 'normal' / f'{i}.png', mesh.render(camera, shader=NormalShader(antialias=True)).visualize().item())
                break

if __name__ == '__main__':
    TaskGroup(
        s4r_air=Tester(
            mesh=Path('data') / 'meshes' / 'air_baloons_shape-300000.ply',
            material=Path('data') / 'materials' / 'airbaloons_material-100000',
            output=Path('data') / 'materials' / 'airbaloons_material-100000',
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'Synthetic4Relight' / 'air_baloons'),
            z_up=False,
            cuda=0,
        ),
        tsir_lego=Tester(
            mesh=Path('data') / 'meshes' / 'tsir_lego_shape-300000.ply',
            material=Path('data') / 'materials' / 'tsir_lego_material-100000',
            output=Path('data') / 'materials' / 'tsir_lego_material-100000',
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'tensoir' / 'lego'),
            z_up=True,
            cuda=0,
        ),
    ).run()
