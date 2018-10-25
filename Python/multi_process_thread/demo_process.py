# 注意Process类的用法
# p.is_alive():判断进程是否还在运行。如果还在运行，返回true，否则返回false
# p.join([timeout]):等待进程实例执行完毕，或等待多少秒
# p.run():默认会调用target指定的对象，如果没有给定target参数，对该进程对象调用start()方法时，就会执行对象中的run()方法
# p.start():启动进程实例(创建子进程）,病运行子进城的run方法
# p.terminate():不管任务是否完成，立即终止,同时不会进行任何的清理工作，如果进程p创建了它自己的子进程，这些进程就会
# 变成僵尸进程，使用时特别注意，如果p保存了一个锁或者参与了进程间通信，那么使用该方法终止它可能会导致死锁或者I/O损坏。
from multiprocessing import Process
import os, time


def test1(interval):
    print('test1子进程运行中，pid=%d， 父进程pid=%d' % (os.getpid(), os.getppid()))
    t_start = time.time()
    time.sleep(interval)
    print('test1执行时间：%.2f秒' % (time.time() - t_start))


def test2(interval):
    print('test2子进程运行中，pid=%d， 父进程pid=%d' % (os.getpid(), os.getppid()))
    t_start = time.time()
    time.sleep(interval)
    print('test2执行时间：%.2f秒' % (time.time() - t_start))


if __name__ == '__main__':
    print('父进程%d' % os.getpid())

    p1 = Process(target=test1, args=(1,))
    p2 = Process(target=test2, name='mark1', args=(2,))

    p1.start()
    p2.start()

    print('p2是否在运行：', p2.is_alive())
    p2.join()   # 等待实例执行完毕，也可以输入等待时间join([timeout])
    print('p2是否在运行：', p2.is_alive())
