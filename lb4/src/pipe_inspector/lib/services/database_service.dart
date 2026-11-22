import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import '../models/inspection.dart';

class DatabaseService {
  static final DatabaseService _instance = DatabaseService._internal();
  factory DatabaseService() => _instance;
  DatabaseService._internal();

  static Database? _database;

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    String path = join(await getDatabasesPath(), 'inspections.db');
    return await openDatabase(
      path,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE inspections(
            id TEXT PRIMARY KEY,
            latitude REAL,
            longitude REAL,
            timestamp TEXT,
            photoPath TEXT,
            material TEXT,
            confidence REAL,
            damagePercent REAL,
            synced INTEGER
          )
        ''');
      },
    );
  }

  Future<void> insertInspection(InspectionRecord record) async {
    final db = await database;
    await db.insert(
      'inspections',
      record.toJson(),
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
    print('Запись сохранена: ${record.id}');
  }

  Future<List<InspectionRecord>> getAllInspections() async {
    final db = await database;
    final List<Map<String, dynamic>> maps = await db.query('inspections');
    return List.generate(maps.length, (i) => InspectionRecord.fromJson(maps[i]));
  }

  Future<List<InspectionRecord>> getUnsyncedInspections() async {
    final db = await database;
    final List<Map<String, dynamic>> maps = await db.query(
      'inspections',
      where: 'synced = ?',
      whereArgs: [0],
    );
    return List.generate(maps.length, (i) => InspectionRecord.fromJson(maps[i]));
  }

  Future<void> markAsSynced(String id) async {
    final db = await database;
    await db.update(
      'inspections',
      {'synced': 1},
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<void> deleteInspection(String id) async {
    final db = await database;
    await db.delete(
      'inspections',
      where: 'id = ?',
      whereArgs: [id],
    );
  }
}