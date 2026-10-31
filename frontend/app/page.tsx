async function getData() {
  const res = await fetch("http://localhost:3000/api/violations", {
    cache: "no-store",
  });

  return res.json();
}

export default async function Page() {
  const data = await getData();

  return (
    <div style={{ padding: 20 }}>
      <h1>🚦 Violation Dashboard</h1>

      {data.length === 0 && <p>No violations yet</p>}

      <div style={{ display: "grid", gap: 20 }}>
        {data.map((v:any, i:any) => (
          <div
            key={i}
            style={{
              border: "1px solid #ccc",
              padding: 10,
              borderRadius: 10,
            }}
          >
            <p><b>ID:</b> {v.id}</p>
            <p><b>Time:</b> {v.time}</p>

            <img
              src={`http://localhost:3000/${v.image}`}
              width={300}
              alt="violation"
            />
          </div>
        ))}
      </div>
    </div>
  );
}