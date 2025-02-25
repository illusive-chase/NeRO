from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type

from rfstudio.engine.task import Task, TaskGroup
from rfstudio.graphics import Cameras, Texture2D, TextureCubeMap, TriangleMesh, RGBAImages, RGBImages
from rfstudio.graphics.shaders import BaseShader, ShadingContext, NormalShader
from rfstudio.io import dump_float32_image, load_float32_image, load_float32_masked_image
from rfstudio.data import RelightDataset
from rfstudio.visualization import TabularFigures
from rfstudio.ui import console
from rfstudio.loss import PSNRLoss, SSIMLoss, LPIPSLoss
from rfstudio.utils.pretty import P

from jaxtyping import Float32
from torch import Tensor
import torch
import numpy as np


@dataclass
class RelightEvaler(Task):

    name: str = ...

    dataset: RelightDataset = RelightDataset(path=...)

    skip_rlit: bool = False

    skip_mat: bool = False

    @torch.no_grad()
    def run(self) -> None:
        (
            gt_albedos,
            gt_roughnesses,
            gt_relights,
            gt_relight_envmaps,
        ) = self.dataset.get_meta(split='test')
        if not self.skip_rlit:
            for relight_idx, relights in enumerate(gt_relights):
                light_name = gt_relight_envmaps[relight_idx].stem
                relight_idx += 1
                psnrs = []
                ssims = []
                lpipss = []
                with console.progress(desc=f'Evaluating Relighting #{relight_idx}') as ptrack:
                    for i in ptrack(range(200)):
                        gt_rgb = relights[i].blend((1, 1, 1)).clamp(0, 1)
                        rgb = RGBAImages([
                            load_float32_masked_image(Path('data') / 'relight' / f'{self.name}-{light_name}' / f'{i:04d}.png')
                        ]).blend((1, 1, 1))
                        psnrs.append(PSNRLoss()(rgb, gt_rgb))
                        ssims.append(1 - SSIMLoss()(rgb, gt_rgb))
                        lpipss.append(LPIPSLoss()(rgb, gt_rgb))
                psnr = torch.stack(psnrs).mean()
                ssim = torch.stack(ssims).mean()
                lpips = torch.stack(lpipss).mean()
                console.print(P@'RLIT[{relight_idx}] @ PSNR: {psnr:.3f}')
                console.print(P@'RLIT[{relight_idx}] @ SSIM: {ssim:.4f}')
                console.print(P@'RLIT[{relight_idx}] @ LPIPS: {lpips:.4f}')
        if not self.skip_mat:
            roughness_mses = []
            psnrs = []
            ssims = []
            lpipss = []
            with console.progress(desc='Evaluating Albedo & Roughness') as ptrack:
                for i in ptrack(range(200)):
                    if gt_roughnesses is not None:
                        roughness = RGBAImages([
                            load_float32_masked_image(
                                Path('data') / 'materials' / f'{self.name}_material-100000' / 'roughness' / f'{i:04d}.png'
                            )
                        ]).blend((0, 0, 0)).item()[..., 0:1] # [H, W, 1]
                        roughness_mses.append(
                            torch.nn.functional.mse_loss(
                                roughness,
                                gt_roughnesses[i].blend((0, 0, 0)).item()[..., 0:1],
                            )
                        )
                    albedo = RGBAImages([
                        load_float32_masked_image(
                            Path('data') / 'materials' / f'{self.name}_material-100000' / 'albedo' / f'{i:04d}.png'
                        )
                    ]).blend((0, 0, 0)).item()
                    gt_albedo = RGBImages([albedo[:, 800:, :]])
                    albedo = RGBImages([albedo[:, :800, :]])
                    psnrs.append(PSNRLoss()(albedo, gt_albedo))
                    ssims.append(1 - SSIMLoss()(albedo, gt_albedo))
                    lpipss.append(LPIPSLoss()(albedo, gt_albedo))
            psnr = torch.stack(psnrs).mean()
            ssim = torch.stack(ssims).mean()
            lpips = torch.stack(lpipss).mean()
            console.print(P@'Albedo @ PSNR: {psnr:.3f}')
            console.print(P@'Albedo @ SSIM: {ssim:.4f}')
            console.print(P@'Albedo @ LPIPS: {lpips:.4f}')
            if gt_roughnesses is not None:
                roughness_mse = torch.stack(roughness_mses).mean()
                console.print(P@'Roughness @ MSE: {roughness_mse:.3f}')
            else:
                console.print(P@'Roughness @ MSE: N/A')


if __name__ == '__main__':
    TaskGroup(
        s4r_air=RelightEvaler(
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'Synthetic4Relight' / 'air_baloons'),
            name='airbaloons',
            cuda=0,
        ),
        s4r_chair=RelightEvaler(
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'Synthetic4Relight' / 'chair'),
            name='chair',
            cuda=0,
        ),
        s4r_hotdog=RelightEvaler(
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'Synthetic4Relight' / 'hotdog'),
            name='hotdog',
            cuda=0,
        ),
        s4r_jugs=RelightEvaler(
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'Synthetic4Relight' / 'jugs'),
            name='jugs',
            cuda=0,
        ),
        tsir_hotdog=RelightEvaler(
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'tensoir' / 'hotdog'),
            name='tsir_hotdog',
            cuda=0,
        ),
        tsir_ficus=RelightEvaler(
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'tensoir' / 'ficus'),
            name='tsir_ficus',
            cuda=0,
        ),
        tsir_arm=RelightEvaler(
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'tensoir' / 'armadillo'),
            name='tsir_arm',
            cuda=0,
        ),
        tsir_lego=RelightEvaler(
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'tensoir' / 'lego'),
            name='tsir_lego',
            cuda=0,
        ),
    ).run()