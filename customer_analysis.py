"""
Advanced Customer Segmentation Analysis
======================================

This script performs comprehensive customer segmentation using K-means clustering
with detailed analysis, insights, and business recommendations.
"""

import random
import math
from collections import defaultdict

def generate_realistic_customer_data(n_customers=1000):
    """Generate realistic customer data with correlations"""
    random.seed(42)
    
    customers = []
    for i in range(n_customers):
        # Create realistic correlations
        age = max(18, min(80, int(random.normalvariate(38, 15))))
        
        # Income correlates with age (older = higher income)
        base_income = 25000 + (age - 18) * 800
        annual_income = max(20000, min(200000, random.normalvariate(base_income, 15000)))
        
        # Spending score correlates with income
        if annual_income > 80000:
            spending_score = random.randint(60, 100)
        elif annual_income > 50000:
            spending_score = random.randint(40, 80)
        else:
            spending_score = random.randint(1, 60)
        
        # Purchase behavior correlates with spending score
        if spending_score > 70:
            total_purchases = random.randint(20, 60)
            avg_order_value = random.uniform(80, 200)
        elif spending_score > 40:
            total_purchases = random.randint(10, 30)
            avg_order_value = random.uniform(40, 120)
        else:
            total_purchases = random.randint(1, 20)
            avg_order_value = random.uniform(10, 80)
        
        # Recency correlates with engagement
        if spending_score > 60:
            days_since_last = random.randint(1, 30)
        elif spending_score > 30:
            days_since_last = random.randint(15, 60)
        else:
            days_since_last = random.randint(30, 180)
        
        # Digital engagement
        if age < 35:
            email_opens = random.randint(8, 25)
            website_visits = random.randint(15, 40)
        else:
            email_opens = random.randint(2, 15)
            website_visits = random.randint(5, 25)
        
        customer = {
            'id': i + 1,
            'age': age,
            'annual_income': round(annual_income),
            'spending_score': spending_score,
            'total_purchases': total_purchases,
            'avg_order_value': round(avg_order_value, 2),
            'days_since_last_purchase': days_since_last,
            'email_opens': email_opens,
            'website_visits': website_visits,
            'lifetime_value': round(total_purchases * avg_order_value, 2)
        }
        customers.append(customer)
    
    return customers

def calculate_distance(customer1, customer2, features):
    """Calculate weighted Euclidean distance between customers"""
    weights = {
        'age': 0.1,
        'annual_income': 0.2,
        'spending_score': 0.25,
        'total_purchases': 0.15,
        'avg_order_value': 0.15,
        'days_since_last_purchase': 0.1,
        'email_opens': 0.05
    }
    
    distance = 0
    for feature in features:
        if feature == 'age':
            val1 = customer1[feature] / 80
            val2 = customer2[feature] / 80
        elif feature == 'annual_income':
            val1 = customer1[feature] / 200000
            val2 = customer2[feature] / 200000
        elif feature == 'spending_score':
            val1 = customer1[feature] / 100
            val2 = customer2[feature] / 100
        elif feature == 'total_purchases':
            val1 = customer1[feature] / 60
            val2 = customer2[feature] / 60
        elif feature == 'avg_order_value':
            val1 = customer1[feature] / 200
            val2 = customer2[feature] / 200
        elif feature == 'days_since_last_purchase':
            val1 = 1 - (customer1[feature] / 180)  # Invert for recency
            val2 = 1 - (customer2[feature] / 180)
        elif feature == 'email_opens':
            val1 = customer1[feature] / 25
            val2 = customer2[feature] / 25
        
        weight = weights.get(feature, 1.0)
        distance += weight * ((val1 - val2) ** 2)
    
    return math.sqrt(distance)

def advanced_kmeans(customers, k=5, max_iterations=50):
    """Advanced K-means with better initialization"""
    features = ['age', 'annual_income', 'spending_score', 'total_purchases', 
                'avg_order_value', 'days_since_last_purchase', 'email_opens']
    
    # Smart initialization: select diverse customers as initial centroids
    centroids = []
    centroids.append(customers[0].copy())
    
    for _ in range(k-1):
        max_distance = 0
        best_customer = None
        
        for customer in customers:
            min_distance_to_centroids = min([
                calculate_distance(customer, centroid, features) 
                for centroid in centroids
            ])
            
            if min_distance_to_centroids > max_distance:
                max_distance = min_distance_to_centroids
                best_customer = customer
        
        if best_customer:
            centroids.append(best_customer.copy())
    
    # K-means iterations
    for iteration in range(max_iterations):
        clusters = [[] for _ in range(k)]
        
        # Assign customers to nearest centroid
        for customer in customers:
            distances = [
                calculate_distance(customer, centroid, features) 
                for centroid in centroids
            ]
            closest_cluster = distances.index(min(distances))
            clusters[closest_cluster].append(customer)
        
        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = {}
                for feature in features:
                    values = [customer[feature] for customer in cluster]
                    new_centroid[feature] = sum(values) / len(values)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[len(new_centroids)])
        
        # Check convergence
        converged = True
        for i in range(k):
            for feature in features:
                if abs(centroids[i][feature] - new_centroids[i][feature]) > 0.01:
                    converged = False
                    break
            if not converged:
                break
        
        centroids = new_centroids
        if converged:
            break
    
    # Assign final cluster labels
    for i, cluster in enumerate(clusters):
        for customer in cluster:
            customer['cluster'] = i
    
    return customers, centroids, clusters

def detailed_cluster_analysis(customers):
    """Perform detailed analysis of each cluster"""
    print("=" * 80)
    print("🎯 ADVANCED CUSTOMER SEGMENTATION ANALYSIS")
    print("=" * 80)
    
    # Group by clusters
    clusters = defaultdict(list)
    for customer in customers:
        clusters[customer['cluster']].append(customer)
    
    # Overall statistics
    total_customers = len(customers)
    total_revenue = sum(customer['lifetime_value'] for customer in customers)
    
    print(f"\n📊 OVERVIEW:")
    print(f"   Total Customers: {total_customers:,}")
    print(f"   Total Revenue: ${total_revenue:,.2f}")
    print(f"   Average Customer Value: ${total_revenue/total_customers:,.2f}")
    
    # Cluster analysis
    print(f"\n🔍 CLUSTER ANALYSIS:")
    print("-" * 80)
    
    features = ['age', 'annual_income', 'spending_score', 'total_purchases', 
                'avg_order_value', 'days_since_last_purchase', 'email_opens', 'website_visits', 'lifetime_value']
    
    cluster_stats = []
    
    for cluster_id in sorted(clusters.keys()):
        cluster_customers = clusters[cluster_id]
        cluster_size = len(cluster_customers)
        cluster_percentage = (cluster_size / total_customers) * 100
        cluster_revenue = sum(customer['lifetime_value'] for customer in cluster_customers)
        cluster_avg_value = cluster_revenue / cluster_size
        
        print(f"\n🏷️  CLUSTER {cluster_id} ({cluster_size} customers - {cluster_percentage:.1f}%)")
        print(f"   Revenue Contribution: ${cluster_revenue:,.2f} ({cluster_revenue/total_revenue*100:.1f}%)")
        print(f"   Average Customer Value: ${cluster_avg_value:,.2f}")
        
        # Calculate averages
        cluster_avg = {}
        for feature in features:
            values = [customer[feature] for customer in cluster_customers]
            cluster_avg[feature] = sum(values) / len(values)
        
        print(f"   📈 Key Metrics:")
        print(f"      • Average Age: {cluster_avg['age']:.1f} years")
        print(f"      • Average Income: ${cluster_avg['annual_income']:,.0f}")
        print(f"      • Spending Score: {cluster_avg['spending_score']:.1f}/100")
        print(f"      • Total Purchases: {cluster_avg['total_purchases']:.1f}")
        print(f"      • Average Order Value: ${cluster_avg['avg_order_value']:.2f}")
        print(f"      • Days Since Last Purchase: {cluster_avg['days_since_last_purchase']:.1f}")
        print(f"      • Email Engagement: {cluster_avg['email_opens']:.1f} opens")
        print(f"      • Website Visits: {cluster_avg['website_visits']:.1f}")
        
        cluster_stats.append({
            'id': cluster_id,
            'size': cluster_size,
            'percentage': cluster_percentage,
            'revenue': cluster_revenue,
            'avg_value': cluster_avg_value,
            'averages': cluster_avg
        })
    
    return cluster_stats, clusters

def generate_business_insights(cluster_stats, clusters):
    """Generate detailed business insights and recommendations"""
    print(f"\n💡 BUSINESS INSIGHTS & STRATEGIC RECOMMENDATIONS")
    print("=" * 80)
    
    # Sort clusters by revenue contribution
    sorted_clusters = sorted(cluster_stats, key=lambda x: x['revenue'], reverse=True)
    
    for i, cluster in enumerate(sorted_clusters):
        cluster_id = cluster['id']
        cluster_customers = clusters[cluster_id]
        avg = cluster['averages']
        
        print(f"\n🎯 CLUSTER {cluster_id} - {cluster['size']} customers (${cluster['revenue']:,.0f} revenue)")
        print("-" * 60)
        
        # Determine segment type and characteristics
        if avg['spending_score'] > 70 and avg['total_purchases'] > 25:
            segment_type = "💎 HIGH-VALUE CHAMPIONS"
            characteristics = [
                "Highest spending and purchase frequency",
                "Strong brand loyalty and engagement",
                "Premium product preferences"
            ]
            recommendations = [
                "🎁 VIP loyalty program with exclusive benefits",
                "📧 Personalized premium product recommendations",
                "🎫 Early access to new products and sales",
                "💬 Direct communication channel with brand"
            ]
        elif avg['annual_income'] > 80000 and avg['spending_score'] > 60:
            segment_type = "💰 AFFLUENT POTENTIAL"
            characteristics = [
                "High income but moderate engagement",
                "Quality over quantity preferences",
                "Price-sensitive but value-conscious"
            ]
            recommendations = [
                "🏆 Premium product showcase and education",
                "💎 Exclusive offers and limited editions",
                "📱 High-touch customer service",
                "🎨 Curated product collections"
            ]
        elif avg['days_since_last_purchase'] > 60:
            segment_type = "⚠️ AT-RISK CUSTOMERS"
            characteristics = [
                "Declining engagement and purchases",
                "Potential churn risk",
                "Need re-engagement strategy"
            ]
            recommendations = [
                "📧 Win-back email campaigns with special offers",
                "🎁 Personalized discount codes",
                "📞 Proactive customer service outreach",
                "🔄 Product recommendation based on history"
            ]
        elif avg['age'] < 35 and avg['website_visits'] > 20:
            segment_type = "📱 DIGITAL NATIVES"
            characteristics = [
                "Young, tech-savvy demographic",
                "High digital engagement",
                "Social media active"
            ]
            recommendations = [
                "📱 Mobile app optimization and features",
                "📸 Social media marketing and influencer partnerships",
                "🎮 Gamification and interactive content",
                "⚡ Fast, seamless digital experience"
            ]
        else:
            segment_type = "🛒 STANDARD CUSTOMERS"
            characteristics = [
                "Average engagement across metrics",
                "Price-conscious shopping behavior",
                "Regular but not frequent purchases"
            ]
            recommendations = [
                "💳 Flexible payment options and financing",
                "📦 Bundle deals and package offers",
                "📧 Regular newsletter with value content",
                "🎯 Targeted promotions based on purchase history"
            ]
        
        print(f"Segment Type: {segment_type}")
        print(f"Key Characteristics:")
        for char in characteristics:
            print(f"   • {char}")
        
        print(f"Strategic Recommendations:")
        for rec in recommendations:
            print(f"   {rec}")
        
        # Calculate potential impact
        current_avg_value = cluster['avg_value']
        if "HIGH-VALUE" in segment_type:
            potential_increase = current_avg_value * 0.15  # 15% increase potential
        elif "AFFLUENT" in segment_type:
            potential_increase = current_avg_value * 0.25  # 25% increase potential
        elif "AT-RISK" in segment_type:
            potential_increase = current_avg_value * 0.20  # 20% retention value
        else:
            potential_increase = current_avg_value * 0.10  # 10% increase potential
        
        print(f"💰 Revenue Potential: ${potential_increase * cluster['size']:,.0f} additional revenue")

def create_summary_report(customers, cluster_stats):
    """Create a comprehensive summary report"""
    print(f"\n📋 EXECUTIVE SUMMARY REPORT")
    print("=" * 80)
    
    total_customers = len(customers)
    total_revenue = sum(customer['lifetime_value'] for customer in customers)
    
    # Top performing clusters
    top_clusters = sorted(cluster_stats, key=lambda x: x['revenue'], reverse=True)[:3]
    
    print(f"🎯 KEY FINDINGS:")
    print(f"   • {total_customers:,} customers analyzed across {len(cluster_stats)} distinct segments")
    print(f"   • Total customer lifetime value: ${total_revenue:,.2f}")
    print(f"   • Top 3 clusters contribute {sum(c['revenue'] for c in top_clusters)/total_revenue*100:.1f}% of total revenue")
    
    print(f"\n🏆 TOP PERFORMING SEGMENTS:")
    for i, cluster in enumerate(top_clusters, 1):
        print(f"   {i}. Cluster {cluster['id']}: {cluster['size']} customers, ${cluster['revenue']:,.0f} revenue")
    
    print(f"\n📈 STRATEGIC PRIORITIES:")
    print(f"   1. Focus retention efforts on high-value segments")
    print(f"   2. Implement targeted re-engagement for at-risk customers")
    print(f"   3. Develop premium offerings for affluent segments")
    print(f"   4. Enhance digital experience for younger demographics")
    
    # Save detailed results
    with open('customer_segments_detailed.csv', 'w') as f:
        header = ['customer_id', 'age', 'annual_income', 'spending_score', 'total_purchases',
                  'avg_order_value', 'days_since_last_purchase', 'email_opens', 'website_visits', 
                  'lifetime_value', 'cluster']
        f.write(','.join(header) + '\n')
        
        for customer in customers:
            row = []
            for field in header:
                if field == 'customer_id':
                    row.append(str(customer.get('id', '')))
                else:
                    row.append(str(customer.get(field, '')))
            f.write(','.join(row) + '\n')
    
    print(f"\n💾 Results saved to: customer_segments_detailed.csv")

def main():
    """Main execution function"""
    print("🚀 Starting Advanced Customer Segmentation Analysis...")
    print("=" * 80)
    
    # Generate realistic customer data
    print("📊 Step 1: Generating realistic customer data...")
    customers = generate_realistic_customer_data(1000)
    print(f"   ✅ Generated {len(customers)} customer records with realistic correlations")
    
    # Perform advanced clustering
    print("🔍 Step 2: Performing advanced K-means clustering...")
    customers, centroids, clusters = advanced_kmeans(customers, k=5)
    print(f"   ✅ Clustering completed with 5 distinct segments")
    
    # Detailed analysis
    print("📈 Step 3: Conducting detailed cluster analysis...")
    cluster_stats, clusters_dict = detailed_cluster_analysis(customers)
    
    # Business insights
    print("💡 Step 4: Generating business insights and recommendations...")
    generate_business_insights(cluster_stats, clusters_dict)
    
    # Summary report
    print("📋 Step 5: Creating executive summary report...")
    create_summary_report(customers, cluster_stats)
    
    print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📊 Analyzed {len(customers)} customers")
    print(f"🎯 Identified {len(cluster_stats)} distinct segments")
    print(f"💰 Total customer value: ${sum(c['lifetime_value'] for c in customers):,.2f}")
    print(f"💾 Detailed results saved to CSV file")

if __name__ == "__main__":
    main()
