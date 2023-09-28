using AsyncIO;
using NetMQ;
using UnityEngine;
using UnityEngine.UI;

public class Main: MonoBehaviour
{
    [Header("[ Update Interval (sec) (FIXED AT AWAKE) ]")]
    [Tooltip("Turn on to fix update interval (sec)")]
    [SerializeField] bool setUpdateInterval = false;
    [SerializeField, Range(0, 5)] float updateInterval = 0;

    [Header("[ Source Fetch & Result Render Resolution (FIXED AT AWAKE) ]")]
    [Tooltip("Resolution to fetch source camera's RenderTexture: Changing this " +
             "does not affect the extent being fetched. Only affects video quality.")]
    [SerializeField] Vector2Int resolution = new Vector2Int(1920, 1080);

    [Header("[ Camera ]")]
    [SerializeField] Camera sourceCamera;

    [Header("[ Result Rendering ]")]
    [SerializeField] GameObject resultPanel;
    [SerializeField] Color resultTint = new Color(255, 255, 255);
    [SerializeField, Range(0, 1)] float resultAlpha = 0.5f;
    RenderTexture camBuffer;
    Texture2D camTexBuffer;
    Texture2D resTexBuffer;
    RawImage resImage;

    [Header("[ Revised DH ]")]
    [SerializeField] GameObject resultPlane;    
    [SerializeField] RawImage resImage_DH;
    

    // [ ZeroMQ ]
    ZMQModule zmq;
    ZMQModule[] zmqModules;
    int zmqModulesSize = 2;
    int zmqIndex = 0;
    float zmqRefreshInterval = 20.0f;


    void Awake()
    {
        // Required to prevent Unity from freezing at 2nd run.
        ForceDotNet.Force();

        // Initialize Global.cs variables
        if(setUpdateInterval)
            Global.Variables.updateInterval = updateInterval;
        else
            Global.Variables.updateInterval = 0;

        // deltaTime Not 0: To run the first iteration
        Global.Variables.deltaTime = Global.Variables.updateInterval + 1;

        // Initialize texture and set it to render the source view
        camBuffer = new RenderTexture(resolution.x, resolution.y, 0);
        sourceCamera.targetTexture = camBuffer;
        camTexBuffer = new Texture2D(resolution.x, resolution.y);

        // Prepare to render the result onto the panel
        resTexBuffer = new Texture2D(resolution.x, resolution.y);
        resImage = resultPanel.GetComponent<RawImage>();

        // Start ZeroMQ
        zmqModules = new ZMQModule[zmqModulesSize];

        for(int i = 0; i < zmqModulesSize; i++)
            zmqModules[i] = new ZMQModule();

        zmqModules[0].Start();
        zmq = zmqModules[0];

        // Assign new ZMQ thread every N(<30)sec interval to prevent freezing
        InvokeRepeating("IterateZMQ", zmqRefreshInterval, zmqRefreshInterval);
    }


    void Update()
    {
        // Fetch new Render Texture at trigger (not a heavy task)
        if(Global.Variables.sourceFetchTrigger)
            FetchSource();

        // Update result image
        if(Global.Variables.resultImgCount > 0)
            UpdateResult();

        // Tint or hide the result panel
        UpdateTint();
    }


    void LateUpdate()
    {
        Global.Variables.accumulatedTime = Time.realtimeSinceStartup;
        Global.Variables.deltaTime += Time.deltaTime;
    }


    void FetchSource()
    {
        // Save current RenderTexture (which would likely be a null)
        RenderTexture current = RenderTexture.active;
        RenderTexture.active = camBuffer;

        // Read RenderTexture
        camTexBuffer.ReadPixels(new Rect(0, 0, resolution.x, resolution.y), 0, 0);
        camTexBuffer.Apply();

        // Encode into a single JPG file to preserve dimension and unify pixel order with stantard images
        Global.Variables.sourceImg = camTexBuffer.EncodeToJPG();

        // Restore current RenderTexture
        RenderTexture.active = current;

        Global.Variables.sourceFetchTrigger = false;
    }


    void UpdateResult()
    {
        // Update the result to the result panel
        resTexBuffer.LoadImage(Global.Variables.resultImg);
        resImage.texture = resTexBuffer;
        resultPlane.GetComponent<Renderer>().material.mainTexture = resTexBuffer; // DH 추가
    }


    void UpdateTint()
    {
        // Update the tint and transparency of the result panel
        if (Global.Variables.resultShow)
        {
            resImage.color = new Color(resultTint.r, resultTint.g, resultTint.b, resultAlpha);
            resImage_DH.color = new Color(resultTint.r, resultTint.g, resultTint.b, resultAlpha); // DH 추가
            resImage_DH.gameObject.SetActive(true);
        }
        else
        {
            resImage.color = new Color(resultTint.r, resultTint.g, resultTint.b, 0);
            resImage_DH.color = new Color(resultTint.r, resultTint.g, resultTint.b, 0); // DH 추가
            resImage_DH.gameObject.SetActive(false);
        }
    }


    void IterateZMQ()
    {
        // Assign new ZMQ thread at every prefixed interval to prevent freezing
        zmqIndex++;

        int prev = (zmqIndex - 1) % zmqModulesSize;
        int next = zmqIndex % zmqModulesSize;

        zmqModules[next].Start();
        zmq = zmqModules[next];

        zmqModules[prev].Stop();
        zmqModules[prev] = new ZMQModule();

        Debug.Log("[INFO]  On ZeroMQ thread #" + next);
    }


    void OnDestroy()
    {
        if(camBuffer != null)
            Destroy(camBuffer);

        if(camTexBuffer != null)
            Destroy(camTexBuffer);

        // Required to prevent Unity from freezing at 2nd run.
        zmq.Stop();
        NetMQConfig.Cleanup();
    }
}
