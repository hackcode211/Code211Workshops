using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class TargetCollision : MonoBehaviour {
    public GameObject txt;
    //public Rigidbody thisthing;
    public float pts;
    //public Vector3 vel;

	// Use this for initialization
	void Start () {
        txt = GameObject.Find("PointText");
        pts = float.Parse(txt.GetComponent<TextMesh>().text);
    }
	
    
    public void OnCollisionEnter(Collision collision)
    {
        //thisthing = gameObject.GetComponent<Rigidbody>();
        string coltype = collision.gameObject.transform.parent.gameObject.name;
        //string str = collision.gameObject.tag;
        
        if (collision.gameObject.transform.parent.gameObject.name == "Walls")
        {
            Destroy(this.gameObject);
        }
        else if (coltype == "Target")
        {
            pts++;
            txt.GetComponent<TextMesh>().text = pts.ToString();
        }
        pushBack();
        //collision.gameObject.GetComponent<Rigidbody>().velocity = Vector3.zero;
        //txt.GetComponent<TextMesh>().text = "did it";


        //pushBack();
    }
    private void pushBack()
    {
        Rigidbody rb = gameObject.GetComponent<Rigidbody>();
        Vector3 vel = rb.velocity;
        rb.velocity = Vector3.zero;
        Vector3 reflforce = Vector3.Reflect(vel, Vector3.down);
        rb.AddForce(Vector3.Scale(reflforce, new Vector3(2, 2, 2)), ForceMode.Impulse);
        //txt2.GetComponent<TextMesh>().text = "success";
    }
}
