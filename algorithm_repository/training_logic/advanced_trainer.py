import copy
import torch
import numpy as np
from ..core_models.base_model import BaseModel
from .base_trainer import BaseTrainer


class AdvancedTrainer(BaseTrainer):
    def __init__(
        self,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 10,
        early_stop_patience: int = 3,
        save_best_model: bool = True
    ):
        """
        初始化训练器。

        :param batch_size: 批大小
        :param learning_rate: 学习率
        :param epochs: 训练轮数
        :param early_stop_patience: 早停轮数 (若 val_loss 连续不上升则停止)
        :param save_best_model: 是否在验证集上表现最好时保存快照
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.save_best_model = save_best_model

    def _split_data(self, X: np.ndarray, Y: np.ndarray, train_ratio=0.7, val_ratio=0.2):
        """
        将数据拆分为训练集和验证集。

        :param X: 特征数组
        :param Y: 目标数组
        :param train_ratio: 训练集比例
        :param val_ratio: 验证集比例
        :return: 分割后的数据集
        """
        total_samples = len(X)
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))

        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        return X_train, Y_train, X_val, Y_val

    def train_model(self, model: BaseModel, X: np.ndarray, Y: np.ndarray):
        """
        训练模型。

        :param model: 模型实例
        :param X: 特征数组
        :param Y: 目标数组
        """
        # 数据拆分
        X_train, Y_train, X_val, Y_val = self._split_data(X, Y)

        # 准备数据加载器
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # 初始化模型
        model.build_model()

        best_val_loss = float('inf')
        best_model_weights = None
        no_improve_count = 0

        print("[INFO] 开始训练 ...")
        for epoch in range(self.epochs):
            model.net.train()
            running_loss = 0.0
            for batch_X, batch_Y in train_loader:
                batch_X, batch_Y = batch_X.to(model.device), batch_Y.to(model.device)

                model.optimizer.zero_grad()
                preds = model.net(batch_X).squeeze(-1)
                loss = model.criterion(preds, batch_Y)
                loss.backward()
                model.optimizer.step()

                running_loss += loss.item() * batch_X.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            # 验证
            model.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_Y in val_loader:
                    batch_X, batch_Y = batch_X.to(model.device), batch_Y.to(model.device)
                    preds = model.net(batch_X).squeeze(-1)
                    loss = model.criterion(preds, batch_Y)
                    val_loss += loss.item() * batch_X.size(0)
            val_loss /= len(val_loader.dataset)

            print(f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")

            # 早停判断
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0

                if self.save_best_model:
                    best_model_weights = copy.deepcopy(model.net.state_dict())
            else:
                no_improve_count += 1
                if no_improve_count >= self.early_stop_patience:
                    print(f"[INFO] 早停触发: {no_improve_count}次未提升，停止训练。")
                    break

        # 恢复最佳模型权重
        if self.save_best_model and best_model_weights is not None:
            model.net.load_state_dict(best_model_weights)
            print("[INFO] 已恢复最佳模型权重。")

        # 保存模型
        model.save_model()
        print("[INFO] 训练完毕！")
