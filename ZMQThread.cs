using System.Threading;

public abstract class ZMQThread
{
    private readonly Thread thread;


    // Create thread first, instead of calling Run() directly.
    // Otherwise it would freeze Unity.
    protected ZMQThread()
    {
        thread = new Thread(Run);
    }


    // Make sure this terminates in finite time period.
    // Use isRunning to determine when to stop the process.
    protected abstract void Run();


    // Set to false when Stop() is called.
    protected bool isRunning {
        get;
        private set;
    }


    public void Start()
    {
        isRunning = true;
        thread.Start();
    }


    public void Stop()
    {
        isRunning = false;

        // Block main thread and wait for the thread to finish first,
        // making sure that the thread ends before the main thread.
        thread.Join();
    }
}
