class InspectionRecord {
  final String id;
  final double latitude;
  final double longitude;
  final DateTime timestamp;
  final String photoPath;
  final String? material;
  final double? confidence;
  final double? damagePercent;
  final bool synced;

  InspectionRecord({
    required this.id,
    required this.latitude,
    required this.longitude,
    required this.timestamp,
    required this.photoPath,
    this.material,
    this.confidence,
    this.damagePercent,
    this.synced = false,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'latitude': latitude,
        'longitude': longitude,
        'timestamp': timestamp.toIso8601String(),
        'photoPath': photoPath,
        'material': material,
        'confidence': confidence,
        'damagePercent': damagePercent,
        'synced': synced ? 1 : 0,
      };

  factory InspectionRecord.fromJson(Map<String, dynamic> json) =>
      InspectionRecord(
        id: json['id'],
        latitude: json['latitude'],
        longitude: json['longitude'],
        timestamp: DateTime.parse(json['timestamp']),
        photoPath: json['photoPath'],
        material: json['material'],
        confidence: json['confidence'],
        damagePercent: json['damagePercent'],
        synced: json['synced'] == 1,
      );

  // Конвертация в GeoJSON
  Map<String, dynamic> toGeoJson() {
    return {
      'type': 'Feature',
      'geometry': {
        'type': 'Point',
        'coordinates': [longitude, latitude],
        'properties': {
          'id': id,
          'timestamp': timestamp.toIso8601String(),
          'material': material,
          'confidence': confidence,
          'damagePercent': damagePercent,
          'source': 'DrainEye',
        },
      },
    };
  }
}