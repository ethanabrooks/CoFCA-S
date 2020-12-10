import multiprocessing
from multiprocessing.queues import Queue as Q


class Queue(Q):
    def __init__(self, *args, **kwargs):
        super().__init__(ctx=multiprocessing.get_context(), *args, **kwargs)
        self.size = 0


def foo(q):
    print(q.size)


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    q = Queue()
    p = multiprocessing.Process(target=foo, args=(q,))
    p.start()
    p.join()
