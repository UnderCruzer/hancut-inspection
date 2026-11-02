#!/usr/bin/env python3
"""Train-only ROI classifier, calib-locked reranking, then one exploratory test evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'ml'))
from hancut.config import load_items
from hancut.data.yolo import class_names
from hancut.eval.roi import assert_disjoint, choose_alpha, covered_fraction, crop_window, proposal_scores


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def patch(image, box, config):
    import cv2
    import numpy as np
    h,w = image.shape[:2]
    x,y,r,b = crop_window(box,w,h,config['context'])
    crop = image[y:b,x:r]
    size = config['crop_size']
    scale = min(size/crop.shape[0],size/crop.shape[1])
    resized = cv2.resize(crop,(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))))
    output = np.full((size,size,3),114,dtype=np.uint8)
    oy,ox = (size-resized.shape[0])//2,(size-resized.shape[1])//2
    output[oy:oy+resized.shape[0],ox:ox+resized.shape[1]] = resized
    return output


def prepare(model, paths, split, data, out, config, names):
    import cv2
    from eval_tiled import read_labels
    rng = random.Random(config['seed'])
    folder = out / 'crops' / split
    folder.mkdir(parents=True, exist_ok=False)
    records = []
    sources = Counter()
    batch = config['detector_batch']
    for start in range(0,len(paths),batch):
        chunk = paths[start:start+batch]
        images = [cv2.imread(str(p)) for p in chunk]
        if any(im is None for im in images):
            raise ValueError('unreadable training image')
        results = model.predict(images,imgsz=640,rect=False,conf=.05,iou=.7,device=0,verbose=False)
        for path, im, result in zip(chunk,images,results,strict=True):
            h,w = im.shape[:2]
            gt = read_labels(data / 'labels' / split / (path.stem+'.txt'),w,h)
            samples = [(row[:4],int(row[4]),'gt') for row in gt]
            def safe(box):
                try:
                    window = crop_window(box,w,h,config['context'])
                except ValueError:
                    return False
                return all(covered_fraction(window,row[:4]) <= .05 for row in gt)
            predictions = result.boxes.data.cpu().numpy()
            negatives = [row[:4] for row in predictions if safe(row[:4])]
            if negatives:
                samples.append((negatives[0],len(names),'hard_negative'))
            else:
                for attempt in range(50):
                    bw,bh = rng.uniform(.08,.35)*w,rng.uniform(.08,.35)*h
                    x,y = rng.uniform(0,w-bw),rng.uniform(0,h-bh)
                    box = (x,y,x+bw,y+bh)
                    if safe(box):
                        samples.append((box,len(names),'random_background'))
                        break
            for j,(box,label,source) in enumerate(samples):
                target = folder / f'{path.stem}_{j}.png'
                if not cv2.imwrite(str(target),patch(im,box,config),[cv2.IMWRITE_PNG_COMPRESSION,1]):
                    raise RuntimeError('crop write failed')
                records.append({'path':str(target),'label':label,'image_id':path.stem,'source':source})
                sources[source] += 1
        if start % (batch*50) == 0 or start+batch >= len(paths):
            status = {'stage':'prepare','split':split,'images':min(start+batch,len(paths)), 'total':len(paths),'crops':len(records)}
            save(out/'status.json',status)
            print(json.dumps(status),flush=True)
    save(out/f'{split}_crops.json',records)
    save(out/f'{split}_crop_counts.json',{'labels':dict(Counter(r['label'] for r in records)),'sources':dict(sources)})
    return records


def transforms(training):
    from torchvision import transforms as T
    steps = [T.RandomHorizontalFlip(),T.RandomVerticalFlip(),T.RandomRotation(15,fill=114),
             T.ColorJitter(brightness=.15,contrast=.15,saturation=.1),T.RandomGrayscale(.15)] if training else []
    return T.Compose(steps+[T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])


def train_classifier(train, val, config, names, out):
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    class Crops(Dataset):
        def __init__(self, records, training):
            self.records,self.transform = records,transforms(training)
        def __len__(self):
            return len(self.records)
        def __getitem__(self,i):
            row = self.records[i]
            with Image.open(row['path']) as image:
                x = self.transform(image.convert('RGB'))
            return x,row['label']

    generator = torch.Generator().manual_seed(config['seed'])
    kwargs = dict(batch_size=config['batch'],num_workers=config['workers'],pin_memory=True,
                  persistent_workers=config['workers']>0)
    training = DataLoader(Crops(train,True),shuffle=True,generator=generator,**kwargs)
    validation = DataLoader(Crops(val,False),shuffle=False,**kwargs)
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features,len(names)+1)
    model.cuda()
    counts = np.bincount([r['label'] for r in train],minlength=len(names)+1)
    if (counts==0).any():
        raise ValueError('training missing a class')
    weights = np.sqrt(counts.sum()/counts)
    weights /= weights.mean()
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device='cuda'))
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['epochs'])
    scaler = torch.amp.GradScaler('cuda')
    best = -1
    history = []
    for epoch in range(config['epochs']):
        tick = time.monotonic()
        # One warm-up epoch trains the new head before adapting pretrained features.
        for parameter in model.features.parameters():
            parameter.requires_grad = epoch > 0
        model.train()
        if epoch == 0:
            model.features.eval()
        total_loss,total = 0.,0
        for step,(x,y) in enumerate(training,1):
            x,y = x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                loss = loss_fn(model(x),y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())*len(y)
            total += len(y)
            if step % 50 == 0:
                progress = {'stage':'training','epoch':epoch+1,'epochs':config['epochs'],
                            'step':step,'steps':len(training),'loss':total_loss/total}
                save(out/'status.json',progress)
                print(json.dumps(progress),flush=True)
        model.eval()
        confusion = torch.zeros((len(names)+1,len(names)+1),dtype=torch.int64,device='cuda')
        with torch.inference_mode():
            for x,y in validation:
                prediction = model(x.cuda(non_blocking=True)).argmax(1)
                pairs = y.cuda(non_blocking=True)*(len(names)+1)+prediction
                confusion += torch.bincount(pairs,minlength=(len(names)+1)**2).reshape(confusion.shape)
        confusion = confusion.cpu().numpy()
        recall = np.diag(confusion)/np.maximum(1,confusion.sum(1))
        score = float(recall[:len(names)].mean())
        row = {'epoch':epoch+1,'loss':total_loss/total,'val_macro_object_recall':score,
               'val_recalls':dict(zip(names+['Background'],map(float,recall))), 'seconds':time.monotonic()-tick}
        history.append(row)
        if score > best:
            best = score
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'names':names+['Background'],'config':config,'epoch':epoch+1},out/'classifier-best.pt')
        save(out/'training.json',history)
        save(out/'status.json',{'stage':'training',**row,'epochs':config['epochs']})
        print(json.dumps(row),flush=True)
        scheduler.step()
    checkpoint = torch.load(out/'classifier-best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()


def refined(full, probabilities, target_ids, alphas, config):
    import numpy as np
    import torch
    from torchvision.ops import batched_nms
    if not any(alphas.values()):
        return full.copy()
    output = [row for i,row in enumerate(full) if int(row[5]) not in target_ids
              or i not in probabilities or alphas[int(row[5])] == 0]
    additions = []
    for i,probs in probabilities.items():
        row = full[i]
        scores = proposal_scores(int(row[5]),float(row[4]),probs,target_ids,alphas)
        for target,score in scores.items():
            if alphas[target] and score >= .001:
                additions.append([*row[:4],score,target])
    candidates = np.array(output+additions,dtype=np.float32).reshape(-1,6)
    mask = np.isin(candidates[:,5],target_ids)
    targets = torch.from_numpy(candidates[mask])
    if len(targets):
        keep = batched_nms(targets[:,:4],targets[:,4],targets[:,5],config['nms_iou'])
        targets = targets[keep]
    return np.concatenate([candidates[~mask],targets.numpy()])


def evaluate(model, classifier, paths, split, data, out, config, names, settings):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from eval_tiled import Metrics, read_labels
    from hancut.eval.scores import rows_for_image, write_predictions
    target_ids = [names.index(n) for n in config['targets']]
    metrics = {arm:Metrics(names,config['fixed_conf']) for arm in settings}
    csv_rows = {arm:[] for arm in settings}
    transform = transforms(False)
    batch = config['detector_batch']
    tick = time.monotonic()
    crop_count = 0
    for start in range(0,len(paths),batch):
        chunk = paths[start:start+batch]
        originals = [cv2.imread(str(p)) for p in chunk]
        results = model.predict(originals,imgsz=640,rect=False,conf=.001,iou=config['nms_iou'],device=0,verbose=False,max_det=300)
        predictions = [r.boxes.data.cpu().numpy() for r in results]
        patches,keys = [],[]
        for j,(im,full) in enumerate(zip(originals,predictions,strict=True)):
            chosen = [i for i in np.argsort(-full[:,4]) if full[i,4] >= config['proposal_min_conf']][:config['max_proposals']]
            for i in chosen:
                x = patch(im,full[i,:4],config)
                patches.append(transform(Image.fromarray(cv2.cvtColor(x,cv2.COLOR_BGR2RGB))))
                keys.append((j,i))
        probs = [{} for _ in chunk]
        with torch.inference_mode():
            for k in range(0,len(patches),config['batch']):
                p = classifier(torch.stack(patches[k:k+config['batch']]).cuda()).softmax(1).cpu().tolist()
                for (j,i),values in zip(keys[k:k+config['batch']],p,strict=True):
                    probs[j][i] = values
        crop_count += len(patches)
        for j,(path,im,full) in enumerate(zip(chunk,originals,predictions,strict=True)):
            gt = read_labels(data/'labels'/split/(path.stem+'.txt'),im.shape[1],im.shape[0])
            present = set(map(int,gt[:,4]))
            for arm,alphas in settings.items():
                pred = refined(full,probs[j],target_ids,alphas,config)
                if not np.array_equal(full[~np.isin(full[:,5],target_ids)],pred[~np.isin(pred[:,5],target_ids)]):
                    raise RuntimeError('non-target regression in merge')
                metrics[arm].add(pred,gt)
                csv_rows[arm].extend(rows_for_image(path.stem,[(int(r[5]),float(r[4])) for r in pred],present,names))
        if start % (batch*20) == 0 or start+batch>=len(paths):
            status = {'stage':'evaluate','split':split,'images':min(start+batch,len(paths)),'total':len(paths),'seconds':time.monotonic()-tick}
            save(out/'status.json',status)
            print(json.dumps(status),flush=True)
    report = {'split':split,'seconds':time.monotonic()-tick,'roi_count':crop_count,
              'arms':{arm:m.summary() for arm,m in metrics.items()}}
    save(out/f'metrics-{split}.json',report)
    for arm,rows in csv_rows.items():
        write_predictions(rows,out/'predictions'/arm/f'{split}.csv',fmt='.9g')
    return report


def operating_report(out):
    from hancut.eval import cli, resplit
    from hancut.eval.grouped import grouped_table
    report = {}
    for arm in ('baseline','refined'):
        rows = [r for d in ('easy','hard','hidden') for r in cli.read_predictions(out/'predictions'/arm/f'test_{d}.csv')]
        ids,_ = resplit.split_images(resplit.image_table(rows),.5,0)
        calibration,evaluation = resplit.partition(rows,ids)
        def point(rs,table):
            p = cli.point_for(rs,table)
            return {**p,'n_non_threat':p['n']-p['n_threat']}
        report[arm] = {}
        for cap in (.05,.02,.01):
            table = grouped_table(calibration,cap,'upper',.05/36)
            report[arm][str(cap)] = {'overall':point(evaluation,table),
                'by_item':{n:point([r for r in evaluation if r['item']==n],table) for n in sorted({r['item'] for r in evaluation})}}
    save(out/'e2.json',report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=REPO/'ml/configs/roi_refiner_20261027.json')
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=REPO/'data/yolo')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if digest(args.weights) != config['weights_sha256']:
        raise ValueError('E1 weights mismatch')
    if args.out.exists():
        raise FileExistsError('use a new output folder')
    if args.smoke:
        config.update(epochs=2,workers=2)
    names = class_names(load_items())
    split_paths = {s:sorted((args.data/'images'/s).glob('*.png')) for s in config['expected_counts']}
    assert_disjoint({s:[p.stem for p in paths] for s,paths in split_paths.items()})
    for split,paths in split_paths.items():
        if len(paths) != config['expected_counts'][split]:
            raise ValueError(f'unexpected split count: {split} {len(paths)}')
        if args.smoke:
            split_paths[split] = sorted(random.Random(config['seed']).sample(paths,min(len(paths),256)))
    args.out.mkdir(parents=True)
    save(args.out/'config.json',{**config,'smoke':args.smoke})
    save(args.out/'split_ids.json',{s:[p.stem for p in paths] for s,paths in split_paths.items()})
    import numpy as np
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    save(args.out/'runtime.json',{'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__})
    model = YOLO(str(args.weights))
    if [model.names[i] for i in sorted(model.names)] != names:
        raise ValueError('class order mismatch')
    train = prepare(model,split_paths['train'],'train',args.data,args.out,config,names)
    val = prepare(model,split_paths['val'],'val',args.data,args.out,config,names)
    classifier = train_classifier(train,val,config,names,args.out)
    target_ids = [names.index(n) for n in config['targets']]
    settings = {str(a):{i:a for i in target_ids} for a in config['alphas']}
    calibration = evaluate(model,classifier,split_paths['calib'],'calib',args.data,args.out,config,names,settings)
    selected = {}
    for n,i in zip(config['targets'],target_ids):
        baseline = calibration['arms']['0.0']['per_class'][n]
        selected[i] = choose_alpha(baseline,[(a,calibration['arms'][str(a)]['per_class'][n]) for a in config['alphas']],config['gate'])
    save(args.out/'selected.json',{'source':'calib only','alphas':{names[i]:v for i,v in selected.items()},
         'classifier_sha256':digest(args.out/'classifier-best.pt'),'locked_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    if any(selected.values()):
        settings = {'baseline':{i:0. for i in target_ids},'refined':selected}
        for split in ('test_easy','test_hard','test_hidden'):
            evaluate(model,classifier,split_paths[split],split,args.data,args.out,config,names,settings)
        operating_report(args.out)
        stage = 'complete'
    else:
        stage = 'rejected_on_calib'
    if digest(args.weights) != config['weights_sha256']:
        raise RuntimeError('original model changed')
    save(args.out/'status.json',{'stage':stage,'alphas':{names[i]:v for i,v in selected.items()},'smoke':args.smoke})
    print('FINISHED',stage,selected,flush=True)


if __name__ == '__main__':
    main()
