import fs from "fs";
import path from "path";

export async function GET() {
  const filePath = path.join(process.cwd(), "..", "backend", "violations.json");

  let data = [];

  try {
    const file = fs.readFileSync(filePath, "utf-8");
    data = JSON.parse(file);
  } catch (err) {
    data = [];
  }

  return Response.json(data);
}