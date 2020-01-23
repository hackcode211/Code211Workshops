using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CollidesWithWallDestroy : MonoBehaviour
{
    // Start is called before the first frame update

    //void Start()
    //{

    //}


    // Update is called once per frame
    void Update()
    {
        if (x > 0)
        {
            Destroy(thing);
        }
    }
    public float x = 0;
    public GameObject thing;
    private void OnCollisionEnter(Collision collision)
    {
        x++;
        thing = collision.gameObject;
    }

}
