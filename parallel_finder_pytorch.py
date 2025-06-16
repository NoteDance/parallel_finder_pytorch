import multiprocessing
import time
import torch
from torch.utils.data import TensorDataset, DataLoader


class ParallelFinder:
    def __init__(self, model_list):
        self.model_list = model_list
        manager = multiprocessing.Manager()
        self.logs = manager.dict()
        self.logs['best_loss'] = float('inf')
        self.logs['best_loss_model_idx'] = None
        self.logs['best_time'] = float('inf')
        self.logs['best_time_model_idx'] = None
        self.lock = multiprocessing.Lock()

    def _train_single(self, idx, train_data, train_labels, epochs, batch_size,
                      criterion, optimizer, optimizer_params, device_str):
        device = torch.device(device_str[idx])
        model = self.model_list[idx]()
        model.to(device)

        if isinstance(train_data, torch.Tensor):
            x_tensor = train_data
        else:
            x_tensor = torch.tensor(train_data)
        if isinstance(train_labels, torch.Tensor):
            y_tensor = train_labels
        else:
            y_tensor = torch.tensor(train_labels)

        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size)

        criterion = criterion()
        criterion.to(device)
        optimizer = optimizer[idx](model.parameters(), **optimizer_params[idx])

        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0.0
            batch_count = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
            epoch_time = time.time() - epoch_start
            avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')

            if epoch + 1 == epochs:
                with self.lock:
                    self.logs[f'model_{idx}_loss'] = avg_loss
                    self.logs[f'model_{idx}_time'] = epoch_time
                    if avg_loss < self.logs['best_loss']:
                        self.logs['best_loss'] = avg_loss
                        self.logs['best_loss_model_idx'] = idx
                        self.logs['time_for_best_loss'] = epoch_time
                    if epoch_time < self.logs['best_time']:
                        self.logs['best_time'] = epoch_time
                        self.logs['best_time_model_idx'] = idx
                        self.logs['loss_for_best_time'] = avg_loss

    def find(self, train_data, train_labels,
             epochs=1, batch_size=32,
             criterion=None, optimizer=None, optimizer_params=None,
             device_str=None):
        processes = []
        for idx in range(len(self.model_list)):
            p = multiprocessing.Process(
                target=self._train_single,
                args=(idx, train_data, train_labels, epochs, batch_size,
                      criterion, optimizer, optimizer_params, device_str)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        return
