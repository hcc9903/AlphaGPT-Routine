import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import akshare as ak
from datetime import datetime
import glob
import hmac
import hashlib
import base64
import time
import urllib.parse
import json
import requests
import sqlite3

def _get_env(key, default, cast_type=str):
    val = os.environ.get(key)
    if val is None: return default
    if cast_type == bool:
        return val.lower() in ('true', 'yes') # 布尔值支持多种写法
    return cast_type(val)

INDEX_CODE = _get_env('INDEX_CODE', '000001')
START_DATE = _get_env('START_DATE', '20220101') # 训练数据开始：近10年
END_DATE = _get_env('END_DATE', '20270101') # 训练数据结束
BATCH_SIZE = _get_env('BATCH_SIZE', 1024, int)
TRAIN_ITERATIONS = _get_env('TRAIN_ITERATIONS', 100, int)
MAX_SEQ_LEN = _get_env('MAX_SEQ_LEN', 10, int)
COST_RATE = _get_env('COST_RATE', 0.0004, float)
LAST_NDAYS = _get_env('LAST_NDAYS', 42, int)      # 用于展示最近交易日的数量（默认42个交易日，约2个月）
HOLD_PERIOD = _get_env('HOLD_PERIOD', 11, int)     # 持仓周期
FORCE_TRAIN = _get_env('FORCE_TRAIN', False, bool)  # 若为False且存在本地公式，则直接加载；若为True则强制重新训练
ONLY_LONG = _get_env('ONLY_LONG', True, bool)     # 是否仅做多，适配A股市场
BEST_FORMULA = _get_env('BEST_FORMULA', '')       # 环境变量公式

DINGTALK_WEBHOOK = _get_env('DINGTALK_WEBHOOK', '')
DINGTALK_SECRET = _get_env('DINGTALK_SECRET', '')

def send_dingtalk_msg(text):
    if not DINGTALK_WEBHOOK:
        return
    
    url = DINGTALK_WEBHOOK
    if DINGTALK_SECRET:
        timestamp = str(round(time.time() * 1000))
        secret_enc = DINGTALK_SECRET.encode('utf-8')
        string_to_sign = '{}\n{}'.format(timestamp, DINGTALK_SECRET)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote(base64.b64encode(hmac_code))
        url = f"{DINGTALK_WEBHOOK}&timestamp={timestamp}&sign={sign}"

    headers = {'Content-Type': 'application/json'}
    data = {
        "msgtype": "markdown",
        "markdown": {
            "title": "策略信号通知",
            "text": text
        }
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
        print(f"钉钉通知已发送，状态码: {resp.status_code}")
    except Exception as e:
        print(f"发送钉钉通知失败: {e}")

DATA_CACHE_PATH = INDEX_CODE + '_data_cache_final.parquet'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

@torch.jit.script
def _op_jump(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)

@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

OPS_CONFIG = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('GATE', _op_gate, 3),
    ('JUMP', _op_jump, 1),
    ('DECAY', _op_decay, 1),
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('MAX3', lambda x: torch.max(x, torch.max(_ts_delay(x,1), _ts_delay(x,2))), 1)
]

FEATURES = ['RET', 'RET5', 'VOL_CHG', 'V_RET', 'TREND', 'F_BUY_F_REPLAY']

VOCAB = FEATURES + [cfg[0] for cfg in OPS_CONFIG]
VOCAB_SIZE = len(VOCAB)
OP_FUNC_MAP = {i + len(FEATURES): cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
OP_ARITY_MAP = {i + len(FEATURES): cfg[2] for i, cfg in enumerate(OPS_CONFIG)}

class AlphaGPT(nn.Module):
    def __init__(self, d_model=64, n_head=4, n_layer=2):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=128, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)

        self.ln_f = nn.LayerNorm(d_model)
        self.head_actor = nn.Linear(d_model, VOCAB_SIZE)
        self.head_critic = nn.Linear(d_model, 1)

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        last = x[:, -1, :]
        return self.head_actor(last), self.head_critic(last)

class DataEngine:
    def __init__(self):
        pass
    def load(self):
        print(f"正在获取 {INDEX_CODE} 的数据...")

        df = ak.stock_zh_a_hist(symbol=INDEX_CODE, period="daily", start_date=START_DATE, end_date=END_DATE, adjust="qfq")
        if df is None or df.empty:
            try:
                df = ak.index_zh_a_hist(symbol=INDEX_CODE, period="daily", start_date=START_DATE, end_date=END_DATE)
            except:
                pass
        if df is None or df.empty:
            try:
                df = ak.fund_etf_hist_em(symbol=INDEX_CODE, period="daily", start_date=START_DATE, end_date=END_DATE, adjust="qfq")
            except:
                pass
        if df is None or df.empty:
            try:
                df = ak.fund_lof_hist_em(symbol=INDEX_CODE, period="daily", start_date=START_DATE, end_date=END_DATE, adjust="qfq")
            except:
                pass
        if df is None or df.empty:
            raise ValueError("未获取到数据，请检查接口调用或网络是否正常")

        df = df.sort_values('日期').reset_index(drop=True)

        for col in ['开盘', '最高', '最低', '收盘', '成交量']:
            df[col] = pd.to_numeric(df[col], errors='coerce').ffill().bfill()

        self.dates = pd.to_datetime(df['日期'])

        close = df['收盘'].values.astype(np.float32)
        open_ = df['开盘'].values.astype(np.float32)
        high = df['最高'].values.astype(np.float32)
        low = df['最低'].values.astype(np.float32)
        vol = df['成交量'].values.astype(np.float32)

        # 特征因子
        ret = np.zeros_like(close)
        ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-6)
        ret5 = pd.Series(close).pct_change(5).fillna(0).values.astype(np.float32)
        vol_ma = pd.Series(vol).rolling(20).mean().values
        vol_chg = np.zeros_like(vol)
        mask = vol_ma > 0
        vol_chg[mask] = vol[mask] / vol_ma[mask] - 1
        vol_chg = np.nan_to_num(vol_chg).astype(np.float32)
        v_ret = (ret * (vol_chg + 1)).astype(np.float32)
        ma60 = pd.Series(close).rolling(60).mean().values
        trend = np.zeros_like(close)
        mask = ma60 > 0
        trend[mask] = close[mask] / ma60[mask] - 1
        trend = np.nan_to_num(trend).astype(np.float32)
        f_balance,f_buy,f_replay,s_balance = get_margin_balance(INDEX_CODE, pd.to_datetime(df['日期']).dt.strftime('%Y%m%d').tolist())
        f_buy_f_replay = f_buy - f_replay

        open_tensor = torch.from_numpy(open_).to(DEVICE)
        N = open_tensor.shape[0]
        h = HOLD_PERIOD
        ret_mat = torch.full((h - 1, N), -float('inf'), device=DEVICE)
        for k in range(2, h + 1):
            valid_len = N - k
            if valid_len > 0:
                numer = open_tensor[k:]
                denom = open_tensor[1:1 + valid_len] + 1e-6
                arr = (numer - denom) / denom
                ret_mat[k - 2, :valid_len] = arr

        valid_mask = ret_mat != -float('inf')
        pos_mask = (ret_mat > 0) & valid_mask
        neg_mask = (ret_mat < 0) & valid_mask
        any_pos = pos_mask.any(dim=0)
        any_neg = neg_mask.any(dim=0)
        first_pos_idx = torch.argmax(pos_mask.int(), dim=0)
        last_valid_idx = (valid_mask.sum(dim=0) - 1)
        has_valid = last_valid_idx >= 0

        indices = torch.arange(N, device=DEVICE)
        select_long_idx = torch.where(any_pos, first_pos_idx, last_valid_idx.clamp(min=0))
        selected_long = ret_mat[select_long_idx, indices]
        selected_long = torch.where(has_valid, selected_long, torch.zeros_like(selected_long))

        if ONLY_LONG:
            self.target_oto_ret = selected_long
        else:
            select_short_idx = torch.where(any_neg, torch.argmax(neg_mask.int(), dim=0), last_valid_idx.clamp(min=0))
            selected_short = ret_mat[select_short_idx, indices]
            selected_short = torch.where(has_valid, selected_short, torch.zeros_like(selected_short))
            self.target_oto_ret_long = selected_long
            self.target_oto_ret_short = selected_short
            self.target_oto_ret = selected_long

        def robust_norm(x):
            x = x.astype(np.float32)
            median = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - median)) + 1e-6
            res = (x - median) / mad
            return np.clip(res, -5, 5).astype(np.float32)
        
        self.feat_data = torch.stack([
            torch.from_numpy(robust_norm(ret)).to(DEVICE),
            torch.from_numpy(robust_norm(ret5)).to(DEVICE),
            torch.from_numpy(robust_norm(vol_chg)).to(DEVICE),
            torch.from_numpy(robust_norm(v_ret)).to(DEVICE),
            torch.from_numpy(robust_norm(trend)).to(DEVICE),
            f_buy_f_replay
        ])

        self.raw_open = open_tensor
        self.raw_close = torch.from_numpy(close).to(DEVICE)
        self.split_idx = int(len(df) * 0.8)
        print(f"{INDEX_CODE} 数据准备就绪。")
        return self

class DeepQuantMiner:
    def __init__(self, engine):
        self.engine = engine
        self.model = AlphaGPT().to(DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)
        self.best_sharpe = -10.0
        self.best_formula_tokens = None

    def get_strict_mask(self, open_slots, step):
        B = open_slots.shape[0]
        mask = torch.full((B, VOCAB_SIZE), float('-inf'), device=DEVICE)
        remaining_steps = MAX_SEQ_LEN - step
        done_mask = (open_slots == 0)
        mask[done_mask, 0] = 0.0 
        active_mask = ~done_mask
        must_pick_feat = (open_slots >= remaining_steps)
        mask[active_mask, :len(FEATURES)] = 0.0
        can_pick_op_mask = active_mask & (~must_pick_feat)
        if can_pick_op_mask.any():
            mask[can_pick_op_mask, len(FEATURES):] = 0.0
        return mask

    def solve_one(self, tokens):
        stack = []
        try:
            for t in reversed(tokens):
                if t < len(FEATURES):
                    stack.append(self.engine.feat_data[t])
                else:
                    arity = OP_ARITY_MAP[t]
                    if len(stack) < arity: raise ValueError
                    args = [stack.pop() for _ in range(arity)]
                    func = OP_FUNC_MAP[t]
                    if arity == 2: res = func(args[0], args[1])
                    else: res = func(args[0])
                    if torch.isnan(res).any(): res = torch.nan_to_num(res)
                    stack.append(res)
            if len(stack) >= 1:
                final = stack[-1]
                if final.std() < 1e-4: return None
                return final
        except: return None
        return None

    def solve_batch(self, token_seqs):
        B = token_seqs.shape[0]
        results = torch.zeros((B, self.engine.feat_data.shape[1]), device=DEVICE)
        valid_mask = torch.zeros(B, dtype=torch.bool, device=DEVICE)
        for i in range(B):
            res = self.solve_one(token_seqs[i].cpu().tolist())
            if res is not None:
                results[i] = res
                valid_mask[i] = True
        return results, valid_mask
    
    def backtest(self, factors):
        if factors.shape[0] == 0: return torch.tensor([], device=DEVICE)
        split = self.engine.split_idx
        rewards = torch.zeros(factors.shape[0], device=DEVICE)
        for i in range(factors.shape[0]):
            f = factors[i, :split]
            if torch.isnan(f).all() or (f == 0).all() or f.numel() == 0:
                rewards[i] = -2.0
                continue
            sig = torch.tanh(f)
            pos = torch.sign(sig)
            if ONLY_LONG: pos[pos == -1] = 0
            turnover = torch.abs(pos - torch.roll(pos, 1))
            if turnover.numel() > 0: turnover[0] = 0.0
            else:
                rewards[i] = -2.0
                continue
            if ONLY_LONG: target_oto = self.engine.target_oto_ret[:split]
            else:
                long_t = self.engine.target_oto_ret_long[:split]
                short_t = self.engine.target_oto_ret_short[:split]
                target_oto = torch.where(pos == 1, long_t, torch.where(pos == -1, short_t, torch.zeros_like(long_t)))
            pnl = pos * target_oto - turnover * COST_RATE
            try:
                pos_count = int((pos == 1).sum().item())
                if int(pos_count / len(pos) * 100) < 10: reward_score = 0.0
                else:
                    win_count = int(((pos == 1) & (target_oto > 0)).sum().item())
                    win_rate_pct = (win_count / pos_count) * 100.0
                    reward_score = win_rate_pct * pnl.mean().item() * 100.0
            except: reward_score = 0.0
            rewards[i] = torch.tensor(float(reward_score), dtype=torch.float32, device=DEVICE)
        return rewards
    
    def find_best_formula_file(self):
        pattern = f"{INDEX_CODE}_best_formula_*.txt"
        files = glob.glob(pattern)
        if files: return max(files, key=os.path.getctime)
        return None
    
    def load_formula_from_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
                self.best_sharpe = float(lines[0].split(':')[1].strip())
                tokens_str = lines[1].split(':')[1].strip()
                self.best_formula_tokens = [int(x) for x in tokens_str.strip('[]').split(',')]
                print(f"加载本地公式: {filepath} (得分: {self.best_sharpe:.3f})")
                return True
        except: return False
    
    def train(self):
        if BEST_FORMULA:
            encoded_tokens = self.encode(BEST_FORMULA)
            if encoded_tokens:
                self.best_formula_tokens = encoded_tokens
                f_val = self.solve_one(encoded_tokens)
                if f_val is not None:
                    self.best_sharpe = self.backtest(f_val.unsqueeze(0))[0].item()
                return
        if not FORCE_TRAIN:
            formula_file = self.find_best_formula_file()
            if formula_file and self.load_formula_from_file(formula_file): return
        
        print(f"正在挖掘最优策略公式... 最大长度={MAX_SEQ_LEN}")
        pbar = tqdm(range(TRAIN_ITERATIONS))
        for _ in pbar:
            B = BATCH_SIZE
            open_slots = torch.ones(B, dtype=torch.long, device=DEVICE)
            log_probs, tokens = [], []
            curr_inp = torch.zeros((B, 1), dtype=torch.long, device=DEVICE)
            for step in range(MAX_SEQ_LEN):
                logits, _ = self.model(curr_inp)
                mask = self.get_strict_mask(open_slots, step)
                dist = Categorical(logits=(logits + mask))
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                tokens.append(action)
                curr_inp = torch.cat([curr_inp, action.unsqueeze(1)], dim=1)
                arity_tens = torch.zeros(VOCAB_SIZE, dtype=torch.long, device=DEVICE)
                for k,v in OP_ARITY_MAP.items(): arity_tens[k] = v
                op_delta = arity_tens[action] - 1
                delta = torch.where(action >= len(FEATURES), op_delta, torch.full((B,), -1, device=DEVICE))
                delta[open_slots==0] = 0
                open_slots += delta
            seqs = torch.stack(tokens, dim=1)
            with torch.no_grad():
                f_vals, valid_mask = self.solve_batch(seqs)
                valid_idx = torch.where(valid_mask)[0]
                rewards = torch.full((B,), -1.0, device=DEVICE)
                if len(valid_idx) > 0:
                    bt_scores = self.backtest(f_vals[valid_idx])
                    rewards[valid_idx] = bt_scores
                    if bt_scores.max() > self.best_sharpe:
                        self.best_sharpe = bt_scores.max().item()
                        self.best_formula_tokens = seqs[valid_idx[torch.argmax(bt_scores)]].cpu().tolist()
            adv = rewards - rewards.mean()
            loss = -(torch.stack(log_probs, 1).sum(1) * adv).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            pbar.set_postfix({'有效率': f"{len(valid_idx)/B:.1%}", '最高得分': f"{self.best_sharpe:.3f}"})
        self.save_formula()

    def save_formula(self):
        if self.best_formula_tokens is None: return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{INDEX_CODE}_best_formula_{timestamp}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"BestSortino: {self.best_sharpe:.4f}\n")
            f.write(f"Tokens: {self.best_formula_tokens}\n")
            f.write(f"Formula: {self.decode()}\n")
        print(f"公式已保存: {filename}")

    def decode(self, tokens=None):
        if tokens is None: tokens = self.best_formula_tokens
        if tokens is None: return "N/A"
        stream = list(tokens)
        def _parse():
            if not stream: return ""
            t = stream.pop(0)
            if t < len(FEATURES): return FEATURES[t]
            args = [_parse() for _ in range(OP_ARITY_MAP[t])]
            return f"{VOCAB[t]}({','.join(args)})"
        try: return _parse()
        except: return "Invalid"

    def encode(self, formula_str):
        import re
        tokens_raw = re.findall(r'[A-Z0-9_]+|\(|\)|,', formula_str)
        vocab_map = {name: i for i, name in enumerate(VOCAB)}; feat_set = set(FEATURES); pos = 0
        def _parse():
            nonlocal pos
            if pos >= len(tokens_raw): return []
            token = tokens_raw[pos]; pos += 1
            if token in feat_set: return [vocab_map[token]]
            elif token in vocab_map:
                op_idx = vocab_map[token]; arity = OP_ARITY_MAP[op_idx]
                if pos < len(tokens_raw) and tokens_raw[pos] == '(':
                    pos += 1; args_tokens = []
                    for i in range(arity):
                        args_tokens.extend(_parse())
                        if i < arity - 1 and pos < len(tokens_raw) and tokens_raw[pos] == ',': pos += 1
                    if pos < len(tokens_raw) and tokens_raw[pos] == ')': pos += 1
                    return [op_idx] + args_tokens
            return []
        return _parse()

def final_reality_check(miner, engine):
    print("\n" + "="*30)
    print("最终实战模拟 (样本外测试)")
    print("="*30)
    formula_str = miner.decode()
    if miner.best_formula_tokens is None: return
    print(f"策略公式: {formula_str}")
    factor_all = miner.solve_one(miner.best_formula_tokens)
    if factor_all is None: return
    split = engine.split_idx
    test_dates = engine.dates[split:]
    test_factors = factor_all[split:].cpu().numpy()
    position = np.sign(np.tanh(test_factors))
    if ONLY_LONG: position[position == -1] = 0
    if ONLY_LONG: test_ret = engine.target_oto_ret[split:].cpu().numpy()
    else:
        long_t = engine.target_oto_ret_long[split:].cpu().numpy()
        short_t = engine.target_oto_ret_short[split:].cpu().numpy()
        test_ret = np.where(position == 1, long_t, np.where(position == -1, short_t, np.zeros_like(long_t)))
    daily_ret = position * test_ret - np.abs(position - np.roll(position, 1)) * COST_RATE
    equity = (1 + daily_ret).cumprod()
    ann_ret = equity[-1] ** (252/len(equity)) - 1
    vol = np.std(daily_ret) * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / (vol + 1e-6)
    dd = 1 - equity / np.maximum.accumulate(equity)
    max_dd = np.max(dd)

    print(f"测试周期    : {test_dates.iloc[0].date()} ~ {test_dates.iloc[-1].date()}")
    print(f"总回报率    : {equity[-1]-1:.2%}")
    print(f"年化收益率  : {ann_ret:.2%}")
    print(f"年化波动率  : {vol:.2%}")
    print(f"夏普比率    : {sharpe:.3f}")
    print(f"最大回撤    : {max_dd:.2%}")
    print(f"卡玛比率    : {ann_ret/(max_dd+1e-6):.3f}")
    try:
        success_count = int(np.sum(((position == 1) & (test_ret > 0)) | ((position == -1) & (test_ret < 0))))
        total_positions = int(np.sum(position != 0))
        print(f"预测准确率  : {success_count}/{total_positions} = {success_count/total_positions:.1%}")
    except: pass
    print("-" * 60)
    plt.figure(figsize=(12, 6)); plt.plot(test_dates, equity, label='策略收益'); plt.plot(test_dates, (1+test_ret).cumprod(), label='基准(持股不动)', alpha=0.5)
    plt.title(f'实战回测: 年化 {ann_ret:.1%} | 夏普 {sharpe:.3f}'); plt.legend(); plt.savefig('strategy_performance.png')

def show_latest_positions(miner, engine, n_days=5):
    output_lines = []
    def log_print(msg): print(msg); output_lines.append(msg)
    log_print("\n" + "="*40); log_print(f"最近 {n_days} 个交易日持仓信号"); log_print("="*40)
    factor_all = miner.solve_one(miner.best_formula_tokens)
    split = engine.split_idx; test_dates = engine.dates[split:]; test_factors = factor_all[split:].cpu().numpy()
    all_open = engine.raw_open.cpu().numpy(); position = np.sign(np.tanh(test_factors))
    if ONLY_LONG: position[position == -1] = 0
    if ONLY_LONG: test_ret = engine.target_oto_ret[split:].cpu().numpy()
    else: test_ret = np.where(position == 1, engine.target_oto_ret_long[split:].cpu().numpy(), np.where(position == -1, engine.target_oto_ret_short[split:].cpu().numpy(), 0))
    n_display = min(n_days, len(test_dates)); start_idx = len(test_dates) - n_display
    simple_sum_return, compound_equity, investment_count, profit_count = 0.0, 1.0, 0, 0
    markdown_lines = []
    log_print(f"\n{'日期':<12} {'建议仓位':<8} {'预期收益':<10} {'次日开盘':<10} {'离场天数':<8} {'离场日期':<12} {'离场价格':<9}")
    log_print("-" * 90)
    for i in range(start_idx, len(test_dates)):
        date_str = test_dates.iloc[i].strftime('%Y-%m-%d'); pos_v = position[i]
        full_idx = split + i; d1_open = f"{all_open[full_idx+1]:.3f}" if full_idx+1 < len(all_open) else "N/A"
        ret_str, color = "N/A", "#000000"
        exit_date, exit_open, exit_off = "N/A", "N/A", HOLD_PERIOD
        if i < len(test_ret):
            rv = test_ret[i]; simple_sum_return += pos_v * rv
            compound_equity *= (1.0 + pos_v * rv - np.abs(pos_v - (position[i-1] if i>0 else 0)) * COST_RATE)
            if pos_v != 0: 
                investment_count += 1
                if (pos_v == 1 and rv > 0) or (pos_v == -1 and rv < 0): profit_count += 1
            color = "#FF0000" if rv > 0 else "#008000"
            ret_str = f"{rv:+.2%}"
        markdown_lines.append(f"- 📅 {date_str} 持仓: {int(pos_v)} | 收益: <font color=\"{color}\">{ret_str}</font> | 入场: {d1_open}")
        log_print(f"{date_str:<12} {pos_v:<10.0f} {ret_str:<12} {d1_open:<13} {exit_off:<8} {exit_date:<12} {exit_open:<9}")
    log_print("-" * 90); log_print("\n最近统计汇总:")
    if investment_count > 0:
        log_print(f"  总投资次数  : {investment_count}\n  盈利次数    : {profit_count}\n  策略胜率    : {profit_count/investment_count:.2%}")
        log_print(f"  简单总收益  : {simple_sum_return:.2%}\n  复合总收益  : {compound_equity-1:.2%}")
    if DINGTALK_WEBHOOK:
        msg = f"## 📊 策略信号 [{INDEX_CODE}]\n\n" + "\n".join(markdown_lines) + f"\n\n### 📈 汇总\n- **胜率**: {profit_count/max(1,investment_count):.2%}\n- **复合收益**: {compound_equity-1:.2%}"
        send_dingtalk_msg(msg)

def get_margin_balance(stock_code, date_list):
    cache_dir = "margin_balance"
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)
    margin_data, missing_dates = {}, []
    for date in date_list:
        cache_file = os.path.join(cache_dir, f"{date}_margin_data.parquet")
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file); target = df[df['标的证券代码'] == stock_code]
                if not target.empty: margin_data[date] = target.iloc[0].to_dict()
            except: missing_dates.append(date)
        else: missing_dates.append(date)
    today = datetime.today().strftime('%Y%m%d')
    missing_dates = [d for d in missing_dates if d != today]
    if missing_dates:
        new_data = _fetch_margin_data_from_api(stock_code, missing_dates, cache_dir)
        margin_data.update(new_data)
    f_bal, f_buy, f_rep, s_bal = [], [], [], []
    for date in date_list:
        row = margin_data.get(date, {})
        f_bal.append(float(row.get('融资余额', 0))); f_buy.append(float(row.get('融资买入额', 0)))
        f_rep.append(float(row.get('融资偿还额', 0))); s_bal.append(float(row.get('融券余量', 0)))
    return torch.tensor(f_bal, device=DEVICE), torch.tensor(f_buy, device=DEVICE), torch.tensor(f_rep, device=DEVICE), torch.tensor(s_bal, device=DEVICE)

def _fetch_margin_data_from_api(stock_code, date_list, cache_dir):
    margin_data = {}
    for date in tqdm(date_list, desc="获取两融数据"):
        try:
            df = ak.stock_margin_detail_sse(date=date)
            if df is not None and not df.empty:
                df.to_parquet(os.path.join(cache_dir, f"{date}_margin_data.parquet"))
                target = df[df['标的证券代码'] == stock_code]
                if not target.empty:
                    r = target.iloc[0]
                    margin_data[date] = {'融资余额': r['融资余额'], '融资买入额': r['融资买入额'], '融资偿还额': r['融资偿还额'], '融券余量': r['融券余量']}
        except: continue
    return margin_data

def main(realitytest=False):
    eng = DataEngine().load()
    miner = DeepQuantMiner(eng)
    miner.train()
    if realitytest: final_reality_check(miner, eng)
    show_latest_positions(miner, eng, n_days=LAST_NDAYS)

if __name__ == "__main__":
    CODE_FORMULA = _get_env('CODE_FORMULA', '')
    if not CODE_FORMULA: main(realitytest=True)
    else:
        for cf in CODE_FORMULA.split('\n'):
            parts = cf.split(':', 1)
            INDEX_CODE, BEST_FORMULA = parts[0], parts[1]
            main(realitytest=False)
