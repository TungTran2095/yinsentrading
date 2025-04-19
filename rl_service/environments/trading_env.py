"""
Môi trường giao dịch cơ sở cho Reinforcement Learning
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
import requests
import sys
sys.path.append('..')
from config import (
    DATA_SERVICE_URL, DATA_SERVICE_API_PREFIX, MODEL_SERVICE_URL, MODEL_SERVICE_API_PREFIX,
    OBSERVATION_WINDOW, TRANSACTION_FEE, SLIPPAGE, ACTIONS, POSITION_SIZES
)

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Môi trường giao dịch cơ sở cho Reinforcement Learning
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, symbol: str, timeframe: str, initial_balance: float = 10000.0,
                 max_steps: int = 1000, window_size: int = OBSERVATION_WINDOW,
                 commission: float = TRANSACTION_FEE, slippage: float = SLIPPAGE,
                 use_ensemble: bool = True, ensemble_id: Optional[str] = None,
                 data_start_time: Optional[str] = None, data_end_time: Optional[str] = None,
                 render_mode: Optional[str] = None):
        """
        Khởi tạo môi trường giao dịch
        
        Args:
            symbol: Cặp giao dịch (ví dụ: "BTC/USDT")
            timeframe: Khung thời gian (ví dụ: "1h", "1d")
            initial_balance: Số dư ban đầu
            max_steps: Số bước tối đa trong một episode
            window_size: Kích thước cửa sổ quan sát
            commission: Phí giao dịch
            slippage: Trượt giá
            use_ensemble: Sử dụng dự đoán từ ensemble hay không
            ensemble_id: ID của ensemble để lấy dự đoán
            data_start_time: Thời gian bắt đầu dữ liệu
            data_end_time: Thời gian kết thúc dữ liệu
            render_mode: Chế độ hiển thị
        """
        super().__init__()
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_steps = max_steps
        self.window_size = window_size
        self.commission = commission
        self.slippage = slippage
        self.use_ensemble = use_ensemble
        self.ensemble_id = ensemble_id
        self.data_start_time = data_start_time
        self.data_end_time = data_end_time
        self.render_mode = render_mode
        
        # Dữ liệu thị trường
        self.data = None
        self.current_step = 0
        self.current_price = 0
        self.prices = []
        
        # Trạng thái giao dịch
        self.position = 0  # Số lượng tài sản nắm giữ
        self.position_value = 0  # Giá trị vị thế
        self.entry_price = 0  # Giá mua vào
        self.total_pnl = 0  # Tổng lợi nhuận
        self.trades = []  # Lịch sử giao dịch
        
        # Định nghĩa không gian hành động
        # Kết hợp hành động (mua, bán, giữ) và kích thước vị thế
        self.action_space = spaces.Discrete(len(ACTIONS) * len(POSITION_SIZES))
        
        # Định nghĩa không gian quan sát
        # Bao gồm: dữ liệu thị trường, chỉ số kỹ thuật, dự đoán, trạng thái tài khoản
        # Số chiều phụ thuộc vào số lượng đặc trưng trong dữ liệu
        # Sẽ được cập nhật sau khi tải dữ liệu
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, 20), dtype=np.float32
        )
        
        # Tải dữ liệu
        self._load_data()
        
        logger.info(f"Khởi tạo môi trường giao dịch cho {symbol} với timeframe {timeframe}")
    
    def _load_data(self) -> None:
        """
        Tải dữ liệu thị trường từ Data Service
        """
        try:
            # Xây dựng URL
            url = f"{DATA_SERVICE_URL}{DATA_SERVICE_API_PREFIX}/technical/{self.symbol}/{self.timeframe}"
            
            # Xây dựng tham số truy vấn
            params = {}
            if self.data_start_time:
                params["start_time"] = self.data_start_time
            if self.data_end_time:
                params["end_time"] = self.data_end_time
            
            # Thực hiện yêu cầu
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Phân tích phản hồi
            data = response.json()["data"]
            
            if not data:
                raise ValueError(f"Không có dữ liệu cho {self.symbol} với timeframe {self.timeframe}")
            
            # Chuyển đổi thành DataFrame
            self.data = pd.DataFrame(data)
            
            # Chuyển đổi timestamp thành datetime
            self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
            self.data = self.data.set_index("timestamp")
            
            # Lấy dự đoán từ Model Service nếu sử dụng ensemble
            if self.use_ensemble and self.ensemble_id:
                self._load_predictions()
            
            # Cập nhật không gian quan sát dựa trên số lượng đặc trưng
            num_features = self.data.shape[1]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.window_size, num_features), dtype=np.float32
            )
            
            logger.info(f"Đã tải {len(self.data)} dòng dữ liệu cho {self.symbol} với timeframe {self.timeframe}")
        
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu: {e}")
            raise
    
    def _load_predictions(self) -> None:
        """
        Tải dự đoán từ Model Service
        """
        try:
            # Xây dựng URL
            url = f"{MODEL_SERVICE_URL}{MODEL_SERVICE_API_PREFIX}/predict"
            
            # Xây dựng dữ liệu yêu cầu
            data = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "ensemble_id": self.ensemble_id
            }
            
            # Thực hiện yêu cầu
            response = requests.post(url, json=data)
            response.raise_for_status()
            
            # Phân tích phản hồi
            predictions = response.json()
            
            if not predictions:
                logger.warning(f"Không có dự đoán cho {self.symbol} với timeframe {self.timeframe}")
                return
            
            # Chuyển đổi thành DataFrame
            pred_df = pd.DataFrame(predictions)
            
            # Chuyển đổi timestamp thành datetime
            pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
            pred_df = pred_df.set_index("timestamp")
            
            # Thêm dự đoán vào dữ liệu
            self.data = self.data.join(pred_df[["prediction"]], how="left")
            
            # Điền giá trị thiếu
            self.data["prediction"] = self.data["prediction"].fillna(method="ffill")
            
            logger.info(f"Đã tải {len(pred_df)} dự đoán cho {self.symbol} với timeframe {self.timeframe}")
        
        except Exception as e:
            logger.warning(f"Lỗi khi tải dự đoán: {e}")
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Đặt lại môi trường về trạng thái ban đầu
        
        Returns:
            Tuple gồm quan sát ban đầu và thông tin
        """
        super().reset(seed=seed)
        
        # Đặt lại trạng thái
        self.balance = self.initial_balance
        self.current_step = 0
        self.position = 0
        self.position_value = 0
        self.entry_price = 0
        self.total_pnl = 0
        self.trades = []
        
        # Lấy quan sát ban đầu
        observation = self._get_observation()
        
        # Lấy thông tin
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Thực hiện một bước trong môi trường
        
        Args:
            action: Hành động để thực hiện
            
        Returns:
            Tuple gồm quan sát mới, phần thưởng, cờ kết thúc, cờ cắt ngắn, thông tin
        """
        # Kiểm tra xem episode đã kết thúc chưa
        if self.current_step >= len(self.data) - 1:
            # Nếu đã kết thúc, trả về trạng thái cuối cùng
            observation = self._get_observation()
            reward = 0
            terminated = True
            truncated = False
            info = self._get_info()
            return observation, reward, terminated, truncated, info
        
        # Tăng bước hiện tại
        self.current_step += 1
        
        # Lấy giá hiện tại
        self.current_price = self.data.iloc[self.current_step]["close"]
        self.prices.append(self.current_price)
        
        # Giải mã hành động
        action_type, position_size = self._decode_action(action)
        
        # Thực hiện hành động
        reward = self._execute_action(action_type, position_size)
        
        # Lấy quan sát mới
        observation = self._get_observation()
        
        # Kiểm tra xem episode đã kết thúc chưa
        terminated = False
        truncated = False
        
        # Kết thúc nếu đã đạt đến số bước tối đa hoặc hết dữ liệu
        if self.current_step >= self.max_steps - 1 or self.current_step >= len(self.data) - 1:
            terminated = True
            
            # Đóng vị thế nếu còn
            if self.position != 0:
                self._close_position()
        
        # Kết thúc nếu tài khoản âm
        if self.balance <= 0:
            terminated = True
            reward = -1  # Phạt nặng nếu phá sản
        
        # Lấy thông tin
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _decode_action(self, action: int) -> Tuple[str, float]:
        """
        Giải mã hành động từ không gian hành động
        
        Args:
            action: Hành động từ không gian hành động
            
        Returns:
            Tuple gồm loại hành động và kích thước vị thế
        """
        action_idx = action // len(POSITION_SIZES)
        size_idx = action % len(POSITION_SIZES)
        
        action_type = ACTIONS[action_idx]
        position_size = POSITION_SIZES[size_idx]
        
        return action_type, position_size
    
    def _execute_action(self, action_type: str, position_size: float) -> float:
        """
        Thực hiện hành động giao dịch
        
        Args:
            action_type: Loại hành động (mua, bán, giữ)
            position_size: Kích thước vị thế (% của vốn)
            
        Returns:
            Phần thưởng
        """
        reward = 0
        
        if action_type == "buy":
            # Nếu đang có vị thế short, đóng trước
            if self.position < 0:
                reward += self._close_position()
            
            # Mở vị thế long
            if self.position == 0:
                reward += self._open_position(position_size, is_long=True)
        
        elif action_type == "sell":
            # Nếu đang có vị thế long, đóng trước
            if self.position > 0:
                reward += self._close_position()
            
            # Mở vị thế short
            if self.position == 0:
                reward += self._open_position(position_size, is_long=False)
        
        elif action_type == "hold":
            # Không làm gì
            # Tính phần thưởng dựa trên lợi nhuận giả định
            if self.position != 0:
                unrealized_pnl = self._calculate_unrealized_pnl()
                reward = unrealized_pnl / self.initial_balance
        
        return reward
    
    def _open_position(self, position_size: float, is_long: bool) -> float:
        """
        Mở vị thế mới
        
        Args:
            position_size: Kích thước vị thế (% của vốn)
            is_long: True nếu là vị thế long, False nếu là vị thế short
            
        Returns:
            Phần thưởng
        """
        # Tính toán giá trị vị thế
        position_value = self.balance * position_size
        
        # Tính toán số lượng tài sản
        price_with_slippage = self.current_price * (1 + self.slippage) if is_long else self.current_price * (1 - self.slippage)
        position_size_in_asset = position_value / price_with_slippage
        
        # Tính toán phí giao dịch
        fee = position_value * self.commission
        
        # Cập nhật số dư
        self.balance -= fee
        
        # Cập nhật vị thế
        self.position = position_size_in_asset if is_long else -position_size_in_asset
        self.position_value = position_value
        self.entry_price = price_with_slippage
        
        # Ghi lại giao dịch
        trade = {
            "timestamp": self.data.index[self.current_step],
            "action": "buy" if is_long else "sell",
            "price": price_with_slippage,
            "amount": abs(self.position),
            "value": position_value,
            "fee": fee
        }
        self.trades.append(trade)
        
        # Phần thưởng là 0 vì chưa có lợi nhuận
        return 0
    
    def _close_position(self) -> float:
        """
        Đóng vị thế hiện tại
        
        Returns:
            Phần thưởng
        """
        if self.position == 0:
            return 0
        
        # Tính toán giá đóng vị thế
        is_long = self.position > 0
        price_with_slippage = self.current_price * (1 - self.slippage) if is_long else self.current_price * (1 + self.slippage)
        
        # Tính toán giá trị vị thế
        position_value = abs(self.position) * price_with_slippage
        
        # Tính toán phí giao dịch
        fee = position_value * self.commission
        
        # Tính toán lợi nhuận
        if is_long:
            pnl = position_value - self.position_value - fee
        else:
            pnl = self.position_value - position_value - fee
        
        # Cập nhật số dư
        self.balance += position_value - fee
        
        # Cập nhật tổng lợi nhuận
        self.total_pnl += pnl
        
        # Ghi lại giao dịch
        trade = {
            "timestamp": self.data.index[self.current_step],
            "action": "sell" if is_long else "buy",
            "price": price_with_slippage,
            "amount": abs(self.position),
            "value": position_value,
            "fee": fee,
            "pnl": pnl
        }
        self.trades.append(trade)
        
        # Đặt lại vị thế
        self.position = 0
        self.position_value = 0
        self.entry_price = 0
        
        # Phần thưởng là lợi nhuận chia cho số dư ban đầu
        reward = pnl / self.initial_balance
        
        return reward
    
    def _calculate_unrealized_pnl(self) -> float:
        """
        Tính toán lợi nhuận chưa thực hiện
        
        Returns:
            Lợi nhuận chưa thực hiện
        """
        if self.position == 0:
            return 0
        
        is_long = self.position > 0
        
        # Tính toán giá đóng vị thế
        price_with_slippage = self.current_price * (1 - self.slippage) if is_long else self.current_price * (1 + self.slippage)
        
        # Tính toán giá trị vị thế
        position_value = abs(self.position) * price_with_slippage
        
        # Tính toán phí giao dịch
        fee = position_value * self.commission
        
        # Tính toán lợi nhuận
        if is_long:
            pnl = position_value - self.position_value - fee
        else:
            pnl = self.position_value - position_value - fee
        
        return pnl
    
    def _get_observation(self) -> np.ndarray:
        """
        Lấy quan sát hiện tại
        
        Returns:
            Mảng numpy chứa quan sát
        """
        # Lấy cửa sổ dữ liệu
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        
        # Nếu không đủ dữ liệu, pad với dữ liệu đầu tiên
        if end_idx - start_idx < self.window_size:
            padding = self.window_size - (end_idx - start_idx)
            window_data = self.data.iloc[0:1].copy()
            for _ in range(padding - 1):
                window_data = pd.concat([window_data, self.data.iloc[0:1]])
            window_data = pd.concat([window_data, self.data.iloc[start_idx:end_idx]])
        else:
            window_data = self.data.iloc[start_idx:end_idx].copy()
        
        # Thêm thông tin tài khoản
        window_data["balance"] = self.balance
        window_data["position"] = self.position
        window_data["position_value"] = self.position_value
        window_data["entry_price"] = self.entry_price
        window_data["unrealized_pnl"] = self._calculate_unrealized_pnl()
        
        # Chuẩn hóa dữ liệu
        normalized_data = self._normalize_data(window_data)
        
        # Chuyển đổi thành mảng numpy
        observation = normalized_data.values.astype(np.float32)
        
        return observation
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn hóa dữ liệu
        
        Args:
            data: DataFrame chứa dữ liệu
            
        Returns:
            DataFrame đã chuẩn hóa
        """
        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        normalized = data.copy()
        
        # Chuẩn hóa giá
        price_cols = ["open", "high", "low", "close"]
        if all(col in normalized.columns for col in price_cols):
            # Chuẩn hóa theo giá đóng cửa đầu tiên
            first_close = normalized["close"].iloc[0]
            for col in price_cols:
                normalized[col] = normalized[col] / first_close - 1.0
        
        # Chuẩn hóa khối lượng
        if "volume" in normalized.columns:
            max_volume = normalized["volume"].max()
            if max_volume > 0:
                normalized["volume"] = normalized["volume"] / max_volume
        
        # Chuẩn hóa chỉ số kỹ thuật
        for col in normalized.columns:
            if col not in price_cols + ["volume", "timestamp"]:
                # Sử dụng z-score
                mean = normalized[col].mean()
                std = normalized[col].std()
                if std > 0:
                    normalized[col] = (normalized[col] - mean) / std
        
        return normalized
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về trạng thái hiện tại
        
        Returns:
            Dictionary chứa thông tin
        """
        return {
            "balance": self.balance,
            "position": self.position,
            "position_value": self.position_value,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self._calculate_unrealized_pnl(),
            "total_pnl": self.total_pnl,
            "total_trades": len(self.trades),
            "current_step": self.current_step,
            "timestamp": self.data.index[self.current_step] if self.current_step < len(self.data) else None
        }
    
    def render(self) -> None:
        """
        Hiển thị môi trường
        """
        if self.render_mode != "human":
            return
        
        info = self._get_info()
        print(f"Step: {info['current_step']}, Timestamp: {info['timestamp']}")
        print(f"Balance: ${info['balance']:.2f}, Position: {info['position']:.6f}")
        print(f"Current Price: ${info['current_price']:.2f}, Entry Price: ${info['entry_price']:.2f}")
        print(f"Unrealized PnL: ${info['unrealized_pnl']:.2f}, Total PnL: ${info['total_pnl']:.2f}")
        print(f"Total Trades: {info['total_trades']}")
        print("-" * 50)
    
    def close(self) -> None:
        """
        Đóng môi trường
        """
        pass
