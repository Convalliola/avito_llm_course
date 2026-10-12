import torch
import torch.distributed as dist


def _ring_step(send_buffer, recv_buffer, rank, world_size):
    """Отправляет буфер следующему процессу в кольце и принимает буфер от предыдущего."""
    request = dist.isend(send_buffer, dst=(rank + 1) % world_size)
    dist.recv(recv_buffer, src=(rank - 1) % world_size)
    request.wait()


def ring_reduce_scatter(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]

    received = torch.empty_like(chunks[0])
    for step in range(world_size - 1):
        send_index = (rank - step - 1) % world_size
        recv_index = (rank - step - 2) % world_size
        _ring_step(chunks[send_index], received, rank, world_size)
        chunks[recv_index] += received
    return chunks[rank]


def ring_all_gather(shard):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)

    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        recv_index = (rank - step - 1) % world_size
        _ring_step(chunks[send_index], chunks[recv_index], rank, world_size)
    return torch.cat(chunks)


def ring_all_reduce(tensor):
    return ring_all_gather(ring_reduce_scatter(tensor))
