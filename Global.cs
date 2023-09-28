using System.Collections.Generic;

public class Global
{
    private static Global variables;

    private Queue<byte[]> _sourceImg = new Queue<byte[]>();
    private bool _sourceFetchTrigger = true;

    private Queue<byte[]> _resultImg = new Queue<byte[]>();
    private Queue<bool> _resultShow = new Queue<bool>();
    private bool _lastResultShow = false;

    private float _updateInterval;
    private float _deltaTime;
    private float _accumulatedTime;


    public static Global Variables {
        get {
            if(variables == null)
                variables = new Global();

            return variables;
        }
    }


    public byte[] sourceImg {
        get {
            if(_sourceImg.Count > 0)
                return _sourceImg.Dequeue();
            else
                return null;
        }
        set {
            // Since ZMQ thread is slower than the main one,
            // update to the most recent RenderTexture for the next fetch
            _sourceImg.Clear();
            _sourceImg.Enqueue(value);
        }
    }


    public int sourceImgCount {
        get {
            return _sourceImg.Count;
        }
    }


    public bool sourceFetchTrigger {
        get {
            return _sourceFetchTrigger;
        }
        set {
            _sourceFetchTrigger = value;
        }
    }


    public byte[] resultImg {
        get {
            if(_resultImg.Count > 0)
                return _resultImg.Dequeue();
            else
                return null;
        }
        set {
            _resultImg.Enqueue(value);
        }
    }


    public int resultImgCount {
        get {
            return _resultImg.Count;
        }
    }


    public bool resultShow {
        get {
            if (_resultShow.Count > 0) {
                _lastResultShow = _resultShow.Dequeue();
                return _lastResultShow;
            } else
                return _lastResultShow;
        }
        set {
            _resultShow.Enqueue(value);
        }
    }


    public float updateInterval {
        get {
            return _updateInterval;
        }
        set {
            _updateInterval = value;
        }
    }


    public float deltaTime {
        get {
            return _deltaTime;
        }
        set {
            _deltaTime = value;
        }
    }


    public float accumulatedTime {
        get {
            return _accumulatedTime;
        }
        set {
            _accumulatedTime = value;
        }
    }
}
