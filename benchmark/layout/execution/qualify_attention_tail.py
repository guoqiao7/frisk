"""Validate compact strict D64 Attention images at N512 and N513 on one H100."""

import argparse
import ctypes as ct
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from physical_resources import query_physical_resources
from tail_attention_reference import BLOCK, ERROR, NAMES, check_output, geometry, inputs, references


H100S = frozenset({'GPU-7bafb1de-da7b-3bf6-e8cd-2369e0c92767',
                   'GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618'})
ROLES = ('Q', 'K', 'V', 'O', 'scratch')
OFFSETS = ((0,0,0,0), (16,32,48,16), (32,16,0,48))
SANITIZERS = ('memcheck', 'racecheck', 'initcheck', 'synccheck')
SANITIZER = Path('/usr/local/cuda-12.8/compute-sanitizer/compute-sanitizer')
GUARD_BYTES = 16384
OUTPUT_CANARY = np.uint8(0xD3)
UNWRITTEN = np.uint32(0x7FC0A5A5)
PROTOCOL = dict(seed=14261107, fixtures=NAMES, offsets=OFFSETS, errors=ERROR,
                numeric='FP32 QK/max/exp2/complete ordered sum/div, FP16 RNE P, FP32 PV',
                oracle='FP64 stable softmax with FP16 P boundary and FP64 products',
                guards='exact logical BHSD spans, disjoint inputs/output, change physical redzones',
                scratch='nullptr', launch_types=('ordinary','CUDA graph'),
                gpu_timing=False)


def _sha(path):
    state = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            state.update(chunk)
    return state.hexdigest()


def _hash_array(array):
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _publish(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _artifact(path, expected):
    path = Path(path)
    if not path.is_absolute() or not path.is_file() or _sha(path) != expected:
        raise ValueError(f'changed or unavailable native artifact {path}')
    return path


def _check_ptx(text, entry):
    if re.findall(r'\.entry\s+([\w$.]+)', text) != [entry]:
        raise ValueError('expected a single registered PTX entry')
    signature = re.search(r'\.entry\s+' + re.escape(entry) + r'\s*\((.*?)\)', text, re.S)
    params = re.findall(r'\.param\s+\.(?:u64|b64)\s+(?:\.ptr\s+\.global\s+\.align\s+\d+\s+)?([\w$.]+)',
                        signature[1] if signature else '')
    if len(params) != 5 or signature[1].count('.param') != 5 or len(set(params)) != 5:
        raise ValueError('PTX must declare five distinct 64-bit pointer slots')
    scratch = re.escape(params[-1])
    if len(re.findall(r'(?<![\w$.])' + scratch + r'(?![\w$.])', text)) != 1:
        raise ValueError('the fifth scratch ABI slot must be unused')
    if not re.search(r'^\s*\.target\s+sm_90a\b', text, re.M):
        raise ValueError('expected Hopper architecture-specific PTX')
    block = re.search(r'\.reqntid\s+(\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?', text)
    if block and tuple(int(x or 1) for x in block.groups()) != BLOCK:
        raise ValueError('PTX thread geometry disagrees with manifest')
    if re.search(r'\b(?:div\.approx\.f32|rcp\.approx\.f32|[\w.]+\.ftz(?:\.[\w.]+)?)(?![\w.])', text):
        raise ValueError('strict trial requires no approximate FP32 division or FTZ')
    return params


def load_manifest(path, case=None, *, allow_partial=False):
    doc = json.loads(Path(path).read_bytes())
    rows = doc['cases']
    if not isinstance(rows, list) or not rows or len({row['case'] for row in rows}) != len(rows):
        raise ValueError('expected a nonempty unique case inventory')
    selected = []
    for row in rows:
        name, spec, image = (row[key] for key in ('case','spec','image'))
        if case is not None and name != case:
            continue
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('invalid registered case name')
        shape, grid = geometry(spec)
        if image['grid'] != grid or image['block'] != list(BLOCK) or image['alignment_bytes'] != 16:
            raise ValueError('manifest launch geometry disagrees with compact BHSD shape')
        dynamic = image['dynamic_shared_bytes']
        if type(dynamic) is not int or not 0 <= dynamic <= 232448:
            raise ValueError('invalid dynamic Shared request')
        binary = _artifact(image['cubin'], image['cubin_sha256'])
        with binary.open('rb') as stream:
            if stream.read(4) != b'\x7fELF':
                raise ValueError('expected an ELF CUBIN')
        ptx = _artifact(image['ptx'], image['ptx_sha256'])
        params = _check_ptx(ptx.read_text(), image['entry'])
        selected.append(dict(row, spans=[int(np.prod(shape))*2]*3+[int(np.prod(shape))*4],
                             ptx_parameters=params))
    if not selected:
        raise ValueError('requested case is absent')
    if case is None and not allow_partial:
        by_batch_heads = {}
        for row in selected:
            spec = row['spec']
            by_batch_heads.setdefault((spec['B'],spec['H']), set()).add(spec['N'])
        if not any(b>1 and h>1 and ns=={512,513} for (b,h),ns in by_batch_heads.items()):
            raise ValueError('qualification requires a paired N512/N513 inventory with B,H both >1')
    return selected


def _smi(*args):
    return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=10).strip()


def gpu_state(uuid):
    card = _smi('-i', uuid, '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader')
    fields = [field.strip() for field in card.split(',')]
    if len(fields) != 4 or fields[1] != uuid or 'H100' not in fields[2]:
        raise ValueError('nvidia-smi did not identify the scheduled H100')
    data = _smi('--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits')
    pids = set()
    for line in data.splitlines():
        device, pid = map(str.strip, line.split(',', 1))
        if device == uuid:
            pids.add(int(pid))
    foreign = sorted(pids-{os.getpid()})
    if foreign:
        raise RuntimeError(f'scheduled {uuid} has foreign GPU processes {foreign}')
    return dict(index=int(fields[0]), uuid=uuid, name=fields[2], driver=fields[3], own_context=bool(pids))


class Driver:
    def __init__(self, uuid):
        if uuid not in H100S or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
            raise ValueError('set CUDA_VISIBLE_DEVICES to the scheduled H100 UUID')
        import torch

        self.torch = torch
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9,0):
            raise ValueError('exactly one visible Hopper device required')
        torch.cuda.set_device(0)
        torch.empty((1,), device='cuda')
        self.cuda = ct.CDLL('libcuda.so.1')
        p = ct.c_void_p
        declarations = {
            'cuCtxGetDevice': [ct.POINTER(ct.c_int)],
            'cuDeviceGetUuid_v2': [p,ct.c_int],
            'cuModuleLoadData': [ct.POINTER(p),p],
            'cuModuleGetFunction': [ct.POINTER(p),p,ct.c_char_p],
            'cuModuleUnload': [p],
            'cuFuncGetParamInfo': [p,ct.c_size_t,ct.POINTER(ct.c_size_t),ct.POINTER(ct.c_size_t)],
            'cuFuncSetAttribute': [p,ct.c_int,ct.c_int],
            'cuLaunchKernel': [p]+[ct.c_uint]*7+[p,ct.POINTER(p),p],
            'cuGraphGetNodes': [p,p,ct.POINTER(ct.c_size_t)],
        }
        for name, signature in declarations.items():
            function = getattr(self.cuda, name)
            function.argtypes, function.restype = signature, ct.c_int
        device, raw = ct.c_int(), (ct.c_ubyte*16)()
        self.call('cuCtxGetDevice', ct.byref(device))
        self.call('cuDeviceGetUuid_v2', ct.byref(raw), device)
        actual = bytes(raw).hex()
        if actual != uuid[4:].replace('-', '').lower():
            raise ValueError('CUDA context identity disagrees with scheduled physical device')

    def call(self, name, *arguments):
        status = getattr(self.cuda, name)(*arguments)
        if status:
            raise RuntimeError(f'{name} returned CUDA status {status}')


class Image:
    def __init__(self, driver, case, *, sanitizer=False):
        self.driver = driver
        self.grid = case['image']['grid']
        image = case['image']
        self.shared = image['dynamic_shared_bytes']
        self.module, self.function = ct.c_void_p(), ct.c_void_p()
        binary = _artifact(image['cubin'], image['cubin_sha256']).read_bytes()
        driver.call('cuModuleLoadData', ct.byref(self.module), ct.c_char_p(binary))
        try:
            driver.call('cuModuleGetFunction', ct.byref(self.function), self.module, image['entry'].encode())
            for slot in range(len(ROLES)):
                offset, size = ct.c_size_t(), ct.c_size_t()
                driver.call('cuFuncGetParamInfo', self.function, slot, ct.byref(offset), ct.byref(size))
                if (offset.value, size.value) != (8*slot,8):
                    raise ValueError('loaded function has an unexpected pointer ABI')
            if not sanitizer and driver.cuda.cuFuncGetParamInfo(
                    self.function, 5, ct.byref(ct.c_size_t()), ct.byref(ct.c_size_t())) != 1:
                raise ValueError('loaded function has an unexpected sixth slot')
            if self.shared:
                driver.call('cuFuncSetAttribute', self.function, 8, self.shared)
            self.resources = query_physical_resources(driver.cuda, self.function, BLOCK[0], self.shared)
            if not self.resources['residency']['launchable_by_resources']:
                raise ValueError('kernel cannot launch within the loaded resource limits')
        except BaseException:
            self.close()
            raise

    def launch(self, buffers, stream):
        self.driver.call('cuLaunchKernel', self.function, *self.grid, *BLOCK, self.shared,
                         ct.c_void_p(stream.cuda_stream), buffers.params, None)

    def capture(self, buffers, stream):
        torch = self.driver.torch
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            self.launch(buffers, stream)
        count = ct.c_size_t()
        self.driver.call('cuGraphGetNodes', ct.c_void_p(graph.raw_cuda_graph()), None, ct.byref(count))
        if count.value != 1:
            raise ValueError('graph must contain exactly one tested kernel')
        graph.instantiate()
        return graph

    def close(self):
        self.driver.torch.cuda.synchronize()
        self.driver.call('cuModuleUnload', self.module)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Buffers:
    def __init__(self, torch, values, spans, offsets, *, poison=0, exact=False, uninitialized=False):
        self.torch = torch
        self.shape = values[0].shape
        if exact and offsets != OFFSETS[0]:
            raise ValueError('exact physical allocation requires zero offsets')
        if [value.nbytes for value in values] + [4*values[0].size] != spans:
            raise ValueError('logical ABI spans disagree with the supplied tensors')
        self.starts = tuple(0 if exact else GUARD_BYTES+offset for offset in offsets)
        self.ends = tuple(start+span for start,span in zip(self.starts,spans))
        suffix = 0 if exact else GUARD_BYTES
        self.host_inputs = []
        self.device_inputs = []
        for value, start, end in zip(values,self.starts,self.ends):
            data = np.empty((end+suffix)//2, dtype=np.uint16)
            if poison == 0:
                data.fill(0x7E00)
            else:
                data[::2], data[1::2] = np.uint16(0x7BFF), np.uint16(0xFBFF)
            raw = data.view(np.uint8)
            raw[start:end] = np.ascontiguousarray(value).view(np.uint8).reshape(-1)
            self.host_inputs.append(raw)
            self.device_inputs.append(torch.from_numpy(raw).to('cuda'))
        self.output = torch.empty((self.ends[-1]+suffix,), dtype=torch.uint8, device='cuda')
        if not uninitialized:
            self.reset()
        pointers = tuple(buffer.data_ptr()+start
                         for buffer,start in zip((*self.device_inputs,self.output),self.starts))
        intervals = sorted((pointer,pointer+span) for pointer,span in zip(pointers,spans))
        if (any(pointer % 16 for pointer in pointers) or any(left[1] > right[0]
                for left,right in zip(intervals,intervals[1:]))):
            raise ValueError('misaligned or overlapping runtime buffers')
        self.words = tuple(ct.c_uint64(x) for x in (*pointers,0))
        self.params = (ct.c_void_p*5)(*(ct.addressof(x) for x in self.words))
        torch.cuda.synchronize()

    def reset(self):
        self.output.fill_(int(OUTPUT_CANARY))
        payload = self.output[self.starts[-1]:self.ends[-1]].view(self.torch.int32)
        payload.fill_(int(UNWRITTEN))

    def check(self, stream, expected, name):
        stream.synchronize()
        output = self.output.cpu().numpy()
        begin, end = self.starts[-1], self.ends[-1]
        if not (np.all(output[:begin]==OUTPUT_CANARY) and np.all(output[end:]==OUTPUT_CANARY)):
            raise ValueError('output redzone modified')
        if not all(np.array_equal(device.cpu().numpy(),host)
                   for device,host in zip(self.device_inputs,self.host_inputs)):
            raise ValueError('read-only input or its redzone modified')
        observed = output[begin:end].copy().view(np.float32).reshape(self.shape)
        facts = check_output(observed, expected, name)
        return observed, dict(facts,sha256=_hash_array(observed), guards=True, inputs_unchanged=True)


def qualify_image(driver, case, stream, uuid, sanitizer):
    spec = case['spec']
    fixtures = ('random','last-key-dominant') if sanitizer else NAMES
    offsets = OFFSETS[:1] if sanitizer else OFFSETS
    launches, neutral = [], None
    with Image(driver, case, sanitizer=bool(sanitizer)) as image:
        for index,name in enumerate(fixtures):
            gpu_state(uuid)
            values = inputs(spec,name,PROTOCOL['seed']+index)
            expected = references(values,name)
            digest = None
            for phase in offsets:
                buf = Buffers(driver.torch,values,case['spans'],phase,exact=bool(sanitizer),
                              uninitialized=(sanitizer=='initcheck'))
                image.launch(buf,stream)
                observed,facts = buf.check(stream,expected,name)
                launches.append(dict(fixture=name,offsets=phase,kind='ordinary',**facts))
                if digest is None:
                    digest = facts['sha256']
                    if name == 'neutral-extension':
                        neutral = observed
                elif facts['sha256'] != digest:
                    raise ValueError('ABI-preserving pointer offsets changed output bits')
                if not sanitizer and name=='random' and phase==offsets[0]:
                    graph = image.capture(buf,stream)
                    with driver.torch.cuda.stream(stream):
                        buf.reset()
                        graph.replay()
                    _,replay = buf.check(stream,expected,name)
                    if replay['sha256']!=digest:
                        raise ValueError('CUDA Graph replay changed output bits')
                    launches.append(dict(fixture=name,offsets=phase,kind='graph',**replay))
                if not sanitizer and name=='key-boundary' and phase==offsets[-1]:
                    poison = Buffers(driver.torch,values,case['spans'],phase,poison=1)
                    image.launch(poison,stream)
                    _,isolated = poison.check(stream,expected,name)
                    if isolated['sha256']!=digest:
                        raise ValueError('invalid padded lanes affect the real compact output')
                    launches.append(dict(fixture=name,offsets=phase,kind='redzone-poison',**isolated))
        return dict(case=case['case'],spec=spec,resources=image.resources,
                    compact_spans=case['spans'],cubin_sha256=case['image']['cubin_sha256'],
                    launches=launches,status='passed'),neutral


def compare_neutral(cases, outputs):
    pairs = {}
    for row in cases:
        name, spec = row['case'], row['spec']
        pairs.setdefault((spec['B'],spec['H']), {}).setdefault(spec['N'], []).append(name)
    checks = []
    for group in pairs.values():
        for base in group.get(512,[]):
            for tail in group.get(513,[]):
                prefix = outputs[tail][..., :512, :]
                if not np.array_equal(outputs[base].view(np.uint32), prefix.view(np.uint32)):
                    raise ValueError(f'zero-weight KV extension changed old output bits for {base} and {tail}')
                duplicate = outputs[tail][...,512,:]
                if not np.array_equal(duplicate.view(np.uint32), prefix[...,0,:].view(np.uint32)):
                    raise ValueError(f'last real query differs from its duplicated query 0 for {tail}')
                checks.append(dict(base=base,tail=tail,status='bitwise',sha256=_hash_array(prefix),
                                   duplicate_query_512_matches_0=True))
    return checks


def _run(cases, manifest, output, uuid, worker):
    if worker and os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') != '1':
        raise ValueError('Compute Sanitizer requires non-caching exact cudaMalloc regions')
    before = gpu_state(uuid)
    output.mkdir()
    _publish(output/'registration.json', dict(protocol=PROTOCOL, cases=cases,
              manifest=dict(path=str(manifest),sha256=_sha(manifest)),gpu=uuid,sanitizer=worker,
              sources={str(p):_sha(p) for p in (Path(__file__),
                  Path(__file__).with_name('tail_attention_reference.py'),
                  Path(__file__).with_name('physical_resources.py'))}))
    driver = Driver(uuid)
    stream = driver.torch.cuda.Stream()
    reports, neutral = [], {}
    for case in cases:
        report,result = qualify_image(driver,case,stream,uuid,worker)
        _publish(output/(case['case']+'.json'),report)
        reports.append(report)
        if result is not None:
            neutral[case['case']]=result
    cross_checks = [] if worker else compare_neutral(cases,neutral)
    _publish(output/'results.json',dict(status='passed',reports=reports,
              neutral_key_extension=cross_checks,before=before,after=gpu_state(uuid),
              sanitized=worker,gpu_timing=False))


def _sanitize(args, cases):
    output = args.output
    output.mkdir()
    summary = []
    for tool in SANITIZERS:
        env = dict(os.environ,PYTORCH_NO_CUDA_MEMORY_CACHING='1')
        command = [str(SANITIZER),'--tool',tool,'--error-exitcode','86']
        if tool == 'racecheck':
            command.extend(['--racecheck-report','all'])
        command.extend([sys.executable,'-B',str(Path(__file__).resolve()),'--manifest',str(args.manifest),
                        '--gpu',args.gpu,'--output',str(output/tool),'--sanitizer-worker',tool])
        if args.case is not None:
            command.extend(['--case',args.case])
        log = output/(tool+'.log')
        with log.open('xb') as stream:
            result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=900)
        log_text = log.read_text(errors='replace')
        errors = re.findall(r'ERROR SUMMARY:\s+(\d+) errors?',log_text)
        hazards = re.findall(r'RACECHECK SUMMARY:\s+(\d+) hazards?',log_text)
        clean = bool(errors or hazards) and all(int(count)==0 for count in errors+hazards)
        row = dict(tool=tool,status='passed' if result.returncode==0 and clean else 'failed',
                   exit_code=result.returncode,log=str(log),log_sha256=_sha(log),
                   case_count=len(cases),exact_allocation=True)
        _publish(output/(tool+'.json'),row)
        summary.append(row)
        if row['status']!='passed':
            break
    passed = len(summary)==len(SANITIZERS) and all(row['status']=='passed' for row in summary)
    _publish(output/'summary.json',dict(status='passed' if passed else 'failed',reports=summary,
                                       sanitizer_sha256=_sha(SANITIZER)))
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--gpu',choices=sorted(H100S))
    parser.add_argument('--case')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight',action='store_true',help='validate a partial or full manifest without using CUDA')
    action.add_argument('--sanitizers',action='store_true',help='run four Compute Sanitizer tools on exact compact allocations')
    action.add_argument('--sanitizer-worker',choices=SANITIZERS,help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.manifest = args.manifest.resolve(strict=True)
    cases = load_manifest(args.manifest,args.case,allow_partial=args.preflight)
    if args.preflight:
        print(json.dumps(dict(manifest_sha256=_sha(args.manifest), gpu_launched=False,
              cases=[dict(case=x['case'],spans=x['spans'],grid=x['image']['grid']) for x in cases])))
        return
    if not args.gpu or args.output is None:
        parser.error('GPU validation requires --output and --gpu')
    args.output = args.output.resolve()
    if args.sanitizers:
        _sanitize(args,cases)
    else:
        _run(cases,args.manifest,args.output,args.gpu,args.sanitizer_worker)


if __name__=='__main__':
    main()
