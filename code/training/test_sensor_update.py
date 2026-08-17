# -*- coding: utf-8 -*-
"""Formal absolute-acceleration test for the 7-channel S5-U sensor-update model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from pix2pixHD_sensor_update_model import Config as ModelConfig, Pix2PixHDSeismicGenerator

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = PACKAGE_ROOT / "data" / "processed" / "sensor_update_acceleration"
DATASET_ROOT = FORMAL_ROOT / "dataset_rgb"
MAPPING_PATH = FORMAL_ROOT / "sensor_update_acc_mapping.json"
NPZ_DIR = PACKAGE_ROOT / "data" / "processed" / "response" / "response_npz"
SPLIT_DIR = PACKAGE_ROOT / "config" / "splits"
DIRECTIONS = ("X", "Y", "Z")
KEY_NODE_GROUPS = {
    "pier_top": [143, 144, 145, 146],
    "pier_bottom": [107, 108, 109, 110],
    "pier_base_reference": [105, 106],
    "bearing_upper": [284, 285, 286, 287, 288, 289, 290, 291],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sensor-mode", choices=("real", "zero"), default="real")
    p.add_argument("--run-name", required=True)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--expected-count", type=int, default=78)
    p.add_argument("--plots-per-record", type=int, default=1, choices=(0,1,2))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def read_lines(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def safe_key(index: int, record: str) -> str:
    clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in record)[:52].rstrip("_")
    digest = hashlib.sha1(record.encode("utf-8")).hexdigest()[:8]
    return f"{index:03d}_{clean}_{digest}"


def load_rgb(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    normalized = rgb.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(normalized.transpose(2,0,1).copy()), rgb


def load_mask(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    return torch.from_numpy((mask.astype(np.float32)/255.0)[None,...]), mask


def normalized_to_rgb(x: np.ndarray) -> np.ndarray:
    return np.rint((np.clip(x,-1,1)+1)*127.5).astype(np.uint8)


def error_rgb(error: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(np.abs(error)/scales.reshape(1,1,3),0,1)*255).astype(np.uint8)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    x=np.asarray(a,dtype=np.float64).reshape(-1); y=np.asarray(b,dtype=np.float64).reshape(-1)
    x=x-x.mean(); y=y-y.mean(); den=math.sqrt(float(x@x)*float(y@y))
    return float((x@y)/den) if den>1e-20 else float("nan")


def metrics(a: np.ndarray, b: np.ndarray) -> dict[str,float]:
    x=np.asarray(a,dtype=np.float64); y=np.asarray(b,dtype=np.float64); e=y-x
    true_peak=float(np.max(np.abs(x))); pred_peak=float(np.max(np.abs(y)))
    return {"mae_m_s2":float(np.mean(np.abs(e))),"rmse_m_s2":float(np.sqrt(np.mean(e*e))),"bias_m_s2":float(np.mean(e)),"correlation":correlation(x,y),"true_peak_m_s2":true_peak,"pred_peak_m_s2":pred_peak,"peak_abs_error_m_s2":abs(pred_peak-true_peak),"peak_rel_error_pct":abs(pred_peak-true_peak)/max(true_peak,1e-12)*100.0,"true_std_m_s2":float(np.std(x)),"pred_std_m_s2":float(np.std(y)),"std_ratio":float(np.std(y))/max(float(np.std(x)),1e-12)}


class RunningStats:
    def __init__(self) -> None:
        self.n=np.zeros(3,dtype=np.int64); self.sa=np.zeros(3); self.ss=np.zeros(3); self.sx=np.zeros(3); self.sy=np.zeros(3); self.sxx=np.zeros(3); self.syy=np.zeros(3); self.sxy=np.zeros(3)
    def update(self, truth: np.ndarray, pred: np.ndarray) -> None:
        for d in range(3):
            x=truth[...,d].astype(np.float64).reshape(-1); y=pred[...,d].astype(np.float64).reshape(-1); e=y-x
            self.n[d]+=x.size; self.sa[d]+=np.sum(np.abs(e)); self.ss[d]+=np.sum(e*e); self.sx[d]+=np.sum(x); self.sy[d]+=np.sum(y); self.sxx[d]+=np.sum(x*x); self.syy[d]+=np.sum(y*y); self.sxy[d]+=np.sum(x*y)
    def rows(self, group: str) -> list[dict[str,Any]]:
        out=[]
        for d,name in enumerate(DIRECTIONS):
            n=float(self.n[d]); mx=self.sx[d]/n; my=self.sy[d]/n; cov=self.sxy[d]-n*mx*my; vx=self.sxx[d]-n*mx*mx; vy=self.syy[d]-n*my*my
            corr=cov/math.sqrt(max(vx*vy,1e-30))
            out.append({"evaluation_group":group,"direction":name,"count":int(n),"mae_m_s2":self.sa[d]/n,"rmse_m_s2":math.sqrt(self.ss[d]/n),"correlation":corr,"true_std_m_s2":math.sqrt(max(vx/n,0)),"pred_std_m_s2":math.sqrt(max(vy/n,0)),"std_ratio":math.sqrt(max(vy,0))/max(math.sqrt(max(vx,0)),1e-15)})
        return out


def write_csv(path: Path, rows: list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def save_panel(path: Path, ground: np.ndarray, sensor: np.ndarray, mask: np.ndarray, target: np.ndarray, pred: np.ndarray, err: np.ndarray) -> None:
    panels=[("Ground RGB",ground),("Sparse sensor RGB",sensor),("Sensor mask",np.repeat(mask[...,None],3,axis=2)),("OpenSees target",target),("Prediction",pred),("Absolute physical error",err)]
    h,w=153,1000; title_h=24; canvas=Image.new("RGB",(w,(h+title_h)*len(panels)),"white"); draw=ImageDraw.Draw(canvas); y=0
    for title,array in panels:
        draw.text((8,y+5),title,fill="black"); canvas.paste(Image.fromarray(array.astype(np.uint8),mode="RGB"),(0,y+title_h)); y+=h+title_h
    path.parent.mkdir(parents=True,exist_ok=True); canvas.save(path)


def plot_node(path: Path, record: str, time: np.ndarray, truth: np.ndarray, pred: np.ndarray, node_id: int, row: int, label: str) -> None:
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
    for d,name in enumerate(DIRECTIONS):
        axes[d].plot(time,truth[row,:,d],label=f"OpenSees {name}"); axes[d].plot(time,pred[row,:,d],"--",label=f"Prediction {name}"); axes[d].set_ylabel(f"{name} absolute acceleration (m/s²)"); axes[d].grid(True,alpha=.3); axes[d].legend()
    axes[-1].set_xlabel("Time (s)"); fig.suptitle(f"{record}\n{label}: node {node_id}"); fig.tight_layout(rect=(0,0,1,.96)); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=160); plt.close(fig)


def main() -> None:
    a=parse_args(); set_seed=42; random.seed(set_seed); np.random.seed(set_seed); torch.manual_seed(set_seed)
    mapping=json.loads(MAPPING_PATH.read_text(encoding="utf-8")); sensor_ids=[int(x) for x in mapping["sensor_node_ids"]]; sensor_rows=[int(x) for x in mapping["sensor_node_rows"]]; acc_scales=np.asarray(mapping["absolute_acceleration_scales_m_s2"],dtype=np.float32)
    output=a.output_root or (FORMAL_ROOT/"runs"/a.run_name/"test_results_best")
    if output.exists():
        if a.overwrite: shutil.rmtree(output)
        else: raise FileExistsError(f"Output exists: {output}; use --overwrite")
    dirs={name:output/name for name in ("predicted_npz","comparison_panels","time_history_plots","predicted_rgb","error_rgb")}
    for p in dirs.values(): p.mkdir(parents=True,exist_ok=True)

    records=read_lines(SPLIT_DIR / "test.txt")
    if len(records) != a.expected_count: raise RuntimeError(f"Test split count={len(records)}, expected={a.expected_count}")
    if a.limit is not None: records=records[:max(0,a.limit)]
    device=torch.device(a.device); ckpt=torch.load(a.checkpoint,map_location=device,weights_only=False)
    config_data=dict(ckpt.get("model_config",{})); config_data["input_nc"]=7; config_data["output_nc"]=3
    model=Pix2PixHDSeismicGenerator(ModelConfig(**config_data)).to(device); model.load_state_dict(ckpt["generator"],strict=True); model.eval()
    print(f"Device: {device}\nCheckpoint epoch: {ckpt.get('epoch')}\nRecords: {len(records)}\nSensor mode: {a.sensor_mode}")

    all_stats=RunningStats(); unobs_stats=RunningStats(); record_rows=[]; key_rows=[]
    unobs_mask=np.ones(153,dtype=bool); unobs_mask[sensor_rows]=False
    for index,record in enumerate(records,start=1):
        split_root=DATASET_ROOT/"test"
        ground_t,ground_rgb=load_rgb(split_root/"ground"/f"{record}.png"); sensor_t,sensor_rgb=load_rgb(split_root/"sensor"/f"{record}.png"); mask_t,mask_rgb=load_mask(split_root/"mask"/f"{record}.png")
        sensor_t=sensor_t*mask_t
        if a.sensor_mode=="zero": sensor_t=torch.zeros_like(sensor_t); mask_t=torch.zeros_like(mask_t); sensor_rgb=np.full_like(sensor_rgb,128); mask_rgb=np.zeros_like(mask_rgb)
        condition=torch.cat([ground_t,sensor_t,mask_t],dim=0).unsqueeze(0).to(device)
        with torch.no_grad(): generated,_=model(condition)
        normalized=generated[0].detach().cpu().numpy().transpose(1,2,0); pred_acc=(normalized*acc_scales.reshape(1,1,3)).astype(np.float32); pred_rgb=normalized_to_rgb(normalized)
        with np.load(NPZ_DIR/f"{record}.npz") as data:
            source_acc=np.asarray(data["accel"],dtype=np.float32); time=np.asarray(data["time"],dtype=np.float32); node_ids=np.asarray(data["node_ids"]).astype(int); coords=np.asarray(data["node_coordinates"],dtype=np.float32); input_acc=np.asarray(data["input"],dtype=np.float32)
        source_mode=mapping["source_npz_acceleration_definition"]; target_mode=mapping["target_acceleration_definition"]
        if source_mode==target_mode: truth=source_acc
        elif source_mode=="relative" and target_mode=="absolute": truth=source_acc+input_acc[None,:,:]
        elif source_mode=="absolute" and target_mode=="relative": truth=source_acc-input_acc[None,:,:]
        else: raise ValueError((source_mode,target_mode))
        row_map={int(n):r for r,n in enumerate(node_ids)}
        if [row_map[n] for n in sensor_ids] != sensor_rows: raise RuntimeError(f"{record}: sensor row mismatch")
        error=pred_acc-truth; err_rgb=error_rgb(error,acc_scales); output_key=safe_key(index,record)
        with Image.open(split_root/"target"/f"{record}.png") as im: target_rgb=np.asarray(im.convert("RGB"))
        Image.fromarray(pred_rgb,mode="RGB").save(dirs["predicted_rgb"]/f"{output_key}.png"); Image.fromarray(err_rgb,mode="RGB").save(dirs["error_rgb"]/f"{output_key}.png")
        save_panel(dirs["comparison_panels"]/f"{output_key}.png",ground_rgb,sensor_rgb,mask_rgb,target_rgb,pred_rgb,err_rgb)
        np.savez_compressed(dirs["predicted_npz"]/f"{output_key}.npz",record_name=np.asarray(record),prediction_accel_abs=pred_acc,true_accel_abs=truth,error_accel_abs=error,time=time,node_ids=node_ids,node_coordinates=coords,input_acc=input_acc,sensor_node_ids=np.asarray(sensor_ids),sensor_node_rows=np.asarray(sensor_rows),unobserved_mask=unobs_mask,checkpoint_epoch=np.asarray(int(ckpt.get("epoch",-1))))
        all_stats.update(truth,pred_acc); unobs_stats.update(truth[unobs_mask],pred_acc[unobs_mask])
        all_m=metrics(truth,pred_acc); un_m=metrics(truth[unobs_mask],pred_acc[unobs_mask])
        row={"record":record,"output_key":output_key,"checkpoint_epoch":int(ckpt.get("epoch",-1)),"all_rmse_m_s2":all_m["rmse_m_s2"],"all_correlation":all_m["correlation"],"unobserved_rmse_m_s2":un_m["rmse_m_s2"],"unobserved_mae_m_s2":un_m["mae_m_s2"],"unobserved_correlation":un_m["correlation"],"unobserved_peak_rel_error_pct":un_m["peak_rel_error_pct"]}
        for d,name in enumerate(DIRECTIONS):
            dm=metrics(truth[unobs_mask,:,d],pred_acc[unobs_mask,:,d]); row[f"{name}_unobserved_rmse_m_s2"]=dm["rmse_m_s2"]; row[f"{name}_unobserved_correlation"]=dm["correlation"]; row[f"{name}_unobserved_std_ratio"]=dm["std_ratio"]
        record_rows.append(row)
        for group,ids in KEY_NODE_GROUPS.items():
            for node in ids:
                if node not in row_map: continue
                r=row_map[node]
                for d,name in enumerate(DIRECTIONS):
                    m=metrics(truth[r,:,d],pred_acc[r,:,d]); key_rows.append({"record":record,"output_key":output_key,"group":group,"node_id":node,"node_row":r,"direction":name,**m})
        # Pier relative responses.
        for label,top,bottom in (("pier1_limb_A",143,107),("pier1_limb_B",144,108),("pier2_limb_A",145,109),("pier2_limb_B",146,110)):
            tr=truth[row_map[top]]-truth[row_map[bottom]]; pr=pred_acc[row_map[top]]-pred_acc[row_map[bottom]]
            for d,name in enumerate(DIRECTIONS): key_rows.append({"record":record,"output_key":output_key,"group":"pier_relative","response_name":label,"direction":name,**metrics(tr[:,d],pr[:,d])})
        if a.plots_per_record>=1:
            # Plot the worst unobserved RMSE node, which is valid for the intended reconstruction target.
            node_rmse=np.sqrt(np.mean((pred_acc-truth)**2,axis=(1,2))); node_rmse[sensor_rows]=-1; worst=int(np.argmax(node_rmse)); plot_node(dirs["time_history_plots"]/f"{output_key}_worst_unobserved_node_{node_ids[worst]}.png",record,time,truth,pred_acc,int(node_ids[worst]),worst,"worst unobserved RMSE")
        if a.plots_per_record>=2:
            for node in (143,144,145,146): plot_node(dirs["time_history_plots"]/f"{output_key}_pier_top_{node}.png",record,time,truth,pred_acc,node,row_map[node],"unmeasured pier top")
        print(f"[{index:03d}/{len(records):03d}] {record} | unobs RMSE={un_m['rmse_m_s2']:.6e} m/s^2 corr={un_m['correlation']:.4f}")

    overall_rows=all_stats.rows("all_153_nodes")+unobs_stats.rows("unobserved_148_nodes")
    write_csv(output/"overall_metrics.csv",overall_rows); write_csv(output/"per_record_metrics.csv",record_rows); write_csv(output/"key_engineering_metrics.csv",key_rows)
    summary={"run_name":a.run_name,"sensor_mode":a.sensor_mode,"checkpoint":str(a.checkpoint.resolve()),"checkpoint_epoch":int(ckpt.get("epoch",-1)),"records":len(records),"sensor_node_ids":sensor_ids,"primary_evaluation":"148 unobserved nodes","overall_metrics":overall_rows}
    (output/"test_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); (output/"TEST_EVALUATION_PASS.txt").write_text("S5-U ABSOLUTE-ACCELERATION FORMAL TEST EVALUATION PASS\n",encoding="utf-8")
    print("\nS5-U ABSOLUTE-ACCELERATION FORMAL TEST EVALUATION PASS"); print(f"Output: {output.resolve()}")


if __name__=="__main__": main()
